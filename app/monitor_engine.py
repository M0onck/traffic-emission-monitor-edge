import cv2
cv2.setNumThreads(1)
import numpy as np
import supervision as sv
import time
import traceback
import logging
import multiprocessing as mp
from multiprocessing import shared_memory
from collections import defaultdict

from ui.renderer import LabelData
from infra.time.ntp_sync import TimeSynchronizer
from domain.physics.spatial_analyzer import SpatialAnalyzer
from domain.physics.kinematics_smoother import KinematicsSmoother
from domain.vehicle.physical_filter import PhysicalVehicleFilter
from domain.physics.vsp_calculator import VSPCalculator
from domain.physics.opmode_mapper import OpModeMapper
from perception.vision_pipeline import VisionPipeline
from perception.daemon import perception_worker

logger = logging.getLogger(__name__)

class TrafficMonitorEngine:
    def __init__(self, config, components, frame_callback=None):
        self.cfg = config
        self.comps = components
        self.frame_callback = frame_callback
        self._is_running = False
        self.time_sync = TimeSynchronizer()
        
        # --- 进程间通信 (IPC) ---
        self.frame_w = self.cfg.FRAME_WIDTH
        self.frame_h = self.cfg.FRAME_HEIGHT
        self.frame_shape = (self.frame_h, self.frame_w, 3)
        self.frame_size = int(np.prod(self.frame_shape))
        self.shm_name = f"nee_cam_shm_{int(time.time())}"
        self.shm = None
        self.bbox_queue = mp.Queue(maxsize=3)
        self.stop_event = mp.Event()

        # --- 核心组件引用 (修复 KeyError: 'renderer') ---
        self.registry = components['registry']
        self.db = components['db']
        self.visualizer = components['visualizer']  # 统一使用 visualizer
        self.plate_worker = components.get('plate_worker')
        self.thermal_cam = components.get('thermal_cam')
        self.weather_station = components.get('weather_station')
        self.classifier = components.get('classifier')

        # --- 业务模型 ---
        self.spatial = SpatialAnalyzer()
        self.kinematics = KinematicsSmoother(max_window=15)
        self.box_filter = PhysicalVehicleFilter(self.cfg)
        self.vsp_calc = VSPCalculator(getattr(config, 'physics_params', {}))
        self.opmode_mapper = OpModeMapper(duration_threshold=1.0)
        self.vision_pipeline = VisionPipeline(fps=config.FPS, label_map={
            "car": self.cfg.YOLO_CLASS_CAR, "bus": self.cfg.YOLO_CLASS_BUS, "truck": self.cfg.YOLO_CLASS_TRUCK
        })

        # --- 状态缓存 ---
        self.current_frame_id = 0
        self.active_tracks = {}
        self.plate_cache = {}
        self.plate_retry = {}
        self.current_session_id = None
        self.profile_stats = defaultdict(float)
        self.profile_frames = 0

    def run(self):
        self._is_running = True
        logger.info("[Engine] 启动多进程解耦引擎...")

        # 准备可序列化的配置字典，确保 UI 的动态修改能传给子进程
        config_dict = {
            'VIDEO_PATH': self.cfg.VIDEO_PATH,
            'HEF_PATH': self.cfg.HEF_PATH,
            'POST_SO_PATH': self.cfg.POST_SO_PATH,
            'FRAME_WIDTH': self.cfg.FRAME_WIDTH,
            'FRAME_HEIGHT': self.cfg.FRAME_HEIGHT,
            'USE_CAMERA': self.cfg.USE_CAMERA
        }

        try:
            # 1. 初始化共享内存
            self.shm = shared_memory.SharedMemory(create=True, size=self.frame_size, name=self.shm_name)
            self.shm_array = np.ndarray(self.frame_shape, dtype=np.uint8, buffer=self.shm.buf)
            self.shm_array[:] = 0

            # 2. 启动感知子进程
            self.p_daemon = mp.Process(
                target=perception_worker,
                args=(self.shm_name, self.frame_shape, self.bbox_queue, self.stop_event, config_dict)
            )
            self.p_daemon.daemon = True
            self.p_daemon.start()

            # 3. 启动传感器
            if self.thermal_cam: self.thermal_cam.start()

            # 4. 创建 Session
            start_timestamp = time.time()
            self.current_session_id = f"Task_{time.strftime('%Y%m%d_%H%M%S')}"
            self.db.create_session(self.current_session_id, start_timestamp, "MainRoad")

            last_hailo_data = []
            prev_env_time = 0
            video_initialized = False

            while self._is_running:
                loop_start = time.perf_counter()
                if not self.p_daemon.is_alive(): break

                # --- A. 数据提取 ---
                current_frame = self.shm_array.copy()
                if not self.bbox_queue.empty():
                    last_hailo_data = self.bbox_queue.get_nowait()

                if not current_frame.any():
                    time.sleep(0.01)
                    continue

                # --- B. 动态坐标与传感器轮询 (1Hz) ---
                now = time.time()
                if not video_initialized:
                    self._initialize_geometry(current_frame)
                    video_initialized = True

                if now - prev_env_time >= 1.0:
                    self._poll_environmental_sensors(now)
                    prev_env_time = now

                # --- C. 核心处理 ---
                self.current_frame_id += 1
                detections = self.vision_pipeline.process(current_frame, last_hailo_data)
                self.process_frame(current_frame, detections, self.current_frame_id, frame_timestamp=now)

                # --- D. 渲染分发 ---
                if self.frame_callback:
                    annotated = self.visualizer.render(current_frame, detections, self._prepare_labels(detections, current_frame.shape))
                    self.frame_callback(annotated)

                # 限速 30fps
                elapsed = time.perf_counter() - loop_start
                time.sleep(max(0.001, (1.0/30.0) - elapsed))

        finally:
            self._cleanup()

    def _initialize_geometry(self, frame):
        """动态坐标系适配逻辑"""
        h, w = frame.shape[:2]
        if 'norm_source_points' in self.comps:
            pts = self.comps['norm_source_points'].copy()
            pts[:, 0] *= w
            pts[:, 1] *= h
            self.visualizer.calibration_points = pts.astype(np.int32)
            from perception.math.geometry import ViewTransformer
            transformer = ViewTransformer(pts, self.comps['target_points'])
            self.spatial.set_transformer(transformer)
            logger.info(f"[Engine] 几何坐标系适配完成: {w}x{h}")

    def _poll_environmental_sensors(self, timestamp):
        """气象与热成像采集逻辑"""
        env_data = {}
        if self.weather_station:
            ws = self.weather_station.get_data()
            if ws.get('isOnline'):
                env_data.update({'air_temp': ws['temp'], 'humidity': ws['humidity'], 'pm25_raw': ws['pm25']})
        if self.thermal_cam:
            tf = self.thermal_cam.read()
            if tf is not None:
                env_data['ground_temp'] = float(np.mean(tf[11:13, 15:17]))
        if env_data:
            self.db.insert_env_raw(self.current_session_id, timestamp, env_data)

    def process_frame(self, frame, detections, frame_id, current_fps=0.0, frame_timestamp=0.0):
        """
        单帧处理流水线。
        """
        h, w = frame.shape[:2]
        
        # --- Step 1: 逻辑过滤 ---
        # 为防止识别错误，采用物理与事实先验知识进行过滤
        t_start = time.perf_counter()
        detections = self.box_filter.apply_pixel_filters(detections, frame.shape)
        if self.comps.get('transformer'):
            detections = self.box_filter.correct_classes_by_physics(detections, self.spatial)
        self.profile_stats['03_physics_filter'] += (time.perf_counter() - t_start)

        # --- Step 2: 注册表更新 (Registry Update) ---
        # 提取标定区域的垂直边界 (用于空间加权)
        t_start = time.perf_counter()
        roi_bounds = None
        if self.comps.get('transformer'):
            # 返回的是 (min_y, max_y)，代表画面中 ROI 的最上端和最下端
            roi_bounds = self.comps['transformer'].get_roi_vertical_bounds()

        # 将 roi_bounds 传给 update 方法
        self.registry.update(detections, frame_id, frame_timestamp, None, roi_bounds=roi_bounds)
        self._handle_exits(frame_id, frame_timestamp)
        self.profile_stats['04_registry_update'] += (time.perf_counter() - t_start)
        
        # --- Step 3: 异步车牌分类 ---
        t_start = time.perf_counter()
        if self.ocr_on and self.plate_worker:
            # 1. 投递新任务
            self._dispatch_plate_tasks(frame, frame_id, detections)
            # 2. 收割已完成的结果 (不阻塞)
            self._collect_plate_results()
        self.profile_stats['05_ocr_dispatch'] += (time.perf_counter() - t_start)

        # --- Step 4: 物理轨迹打点与动态死区判定 ---
        t_start = time.perf_counter()
        if self.motion_on and self.comps.get('transformer'):
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

            for tid, raw_point in zip(detections.tracker_id, points):
                if self.spatial.is_in_roi(raw_point):
                    
                    # 1. 空间组件获取物理坐标与宽容度
                    curr_phys = self.spatial.get_physical_point(raw_point)
                    dynamic_tolerance = self.spatial.get_dynamic_tolerance(raw_point)

                    # 2. 运动学逻辑
                    record = self.registry.get_record(tid)
                    trajectory = record.get('trajectory', []) if record else []
                    pseudo_speed = 0.0
                    
                    if trajectory:
                        last_phys_y = trajectory[-1].get('raw_y', curr_phys[1])
                        # 根据预期的 FPS 动态分配备用时间跨度
                        fallback_dt = 1.0 / self.native_fps
                        last_time = trajectory[-1].get('timestamp', frame_timestamp - fallback_dt)
                        
                        if abs(curr_phys[1] - last_phys_y) < max(0.2, dynamic_tolerance):
                            curr_phys[1] = last_phys_y 
                            pseudo_speed = 0.0
                        else:
                            pseudo_speed = abs(curr_phys[1] - last_phys_y) / max(0.001, frame_timestamp - last_time)

                    self.registry.append_kinematics(
                        tid, frame_id, pseudo_speed, 0.0,
                        raw_x=curr_phys[0], raw_y=curr_phys[1],
                        pixel_x=raw_point[0], pixel_y=raw_point[1],  
                        timestamp=frame_timestamp
                    )
        self.profile_stats['06_kinematics'] += (time.perf_counter() - t_start)

        # --- Step 5: 可视化渲染 ---
        annotated_frame = None
        if not getattr(self, 'headless_mode', False):
            t_start = time.perf_counter()
            # 传入所有追踪到的目标
            filtered_detections = detections
            
            # 为真实车辆和正在识别中的车辆生成 UI 标签数据
            label_data_list = self._prepare_labels(filtered_detections, frame.shape)

            # 传递过滤后的数据给视觉渲染器
            annotated_frame = self.visualizer.render(frame, filtered_detections, label_data_list, fps=current_fps)
            self.profile_stats['07_visualizer_render'] += (time.perf_counter() - t_start)
        
        return annotated_frame

    def _dispatch_plate_tasks(self, frame, frame_id, detections):
        """派发车牌识别任务给子进程"""
        img_h, img_w = frame.shape[:2]

        for tid, box in zip(detections.tracker_id, detections.xyxy):
            # 1. 冷却检查防爆栈：防止对同一辆车频繁提交识别请求
            if frame_id - self.plate_retry.get(tid, -999) < self.cfg.OCR_RETRY_COOLDOWN:
                continue

            x1, y1, x2, y2 = map(int, box)
            
            # 2. 动态 Padding (保护低位悬挂的车牌)
            bw = x2 - x1
            bh = y2 - y1
            pad_x = int(bw * 0.05)
            pad_top = int(bh * 0.05)
            pad_bottom = int(bh * 0.20) 
            
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_top)
            crop_x2 = min(img_w, x2 + pad_x)
            crop_y2 = min(img_h, y2 + pad_bottom)
            
            vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            
            if vehicle_crop.size == 0:
                continue

            # 5. 投递最终的优质任务
            if self.plate_worker.push_task(tid, vehicle_crop):
                self.plate_retry[tid] = frame_id

                logger.debug(f"[DEBUG] 成功向子进程投递 TID={tid} 的车牌识别任务")

    def _collect_plate_results(self):
        """非阻塞地从子进程收取计算结果并入库"""
        results = self.plate_worker.get_results()

        if len(results) > 0:
            logger.debug(f"[DEBUG] 主进程从队列收到了 {len(results)} 条车牌结果")
        
        color_thresholds = {
            'green': 0.60,  
            'blue': 0.85,   
            'yellow': 0.75  
        }

        # 只要这辆车历史上有 N 帧被识别为绿/黄，就强制锁定
        VETO_THRESHOLD = 1

        for tid, color_type, conf, rel_landmarks in results:
            # 把车牌坐标存进 UI 缓存
            if tid not in self.plate_cache:
                self.plate_cache[tid] = {'color': color_type, 'rel_landmarks': rel_landmarks}
            else:
                self.plate_cache[tid]['rel_landmarks'] = rel_landmarks

            target_threshold = color_thresholds.get(color_type, self.cfg.OCR_CONF_THRESHOLD)
            
            if conf > target_threshold: 
                # 1. 正常写入历史记录
                self.registry.add_plate_history(tid, color_type, 1.0, conf)
                
                # 2. 少数派锁定逻辑 
                record = self.registry.get_record(tid)
                history = record.get('plate_history', []) if record else []
                
                green_count = sum(1 for h in history if h['color'] == 'green')
                yellow_count = sum(1 for h in history if h['color'] == 'yellow')
                
                # 决定当前展示给 UI 的颜色
                final_color_for_ui = color_type
                if green_count >= VETO_THRESHOLD:
                    final_color_for_ui = 'green'
                elif yellow_count >= VETO_THRESHOLD:
                    final_color_for_ui = 'yellow'

                # 3. 置信度达标后，更新确认的 UI 颜色
                self.plate_cache[tid]['color'] = final_color_for_ui

    def _handle_exits(self, frame_id, current_timestamp):
        """
        处理离场车辆或赖场车辆：执行最终/临时结算、生成报表并入库。
        """
        for tid, record in self.registry.check_exits(frame_id, current_timestamp):
            
            # 判断是否为拥堵强制结算的“切片”数据
            is_continued = record.get('exit_type') == 'continued'

            # Step 1. 解析最终属性 (车牌、细分车型)
            final_plate, final_type_str = self.classifier.resolve_type(
                record['class_id'], record.get('plate_history', [])
            )

            # 少数派一票否决制 (Minority Veto)
            voted_color = "Unknown"
            history = record.get('plate_history', [])
            
            if history:
                green_count = sum(1 for h in history if h['color'] == 'green')
                yellow_count = sum(1 for h in history if h['color'] == 'yellow')
                
                VETO_THRESHOLD = 2
                
                if green_count >= VETO_THRESHOLD:
                    voted_color = 'green'
                elif yellow_count >= VETO_THRESHOLD:
                    voted_color = 'yellow'
                else:
                    # 若完全没有检出绿牌/黄牌，使用众数投票 (取出现次数最多的颜色，通常是蓝牌)
                    colors = [h['color'] for h in history]
                    voted_color = max(set(colors), key=colors.count)
            
            record['final_plate_color'] = voted_color

            # 将车牌颜色映射为能源类型 (新能源绿牌为 Electric，其他为 Normal)
            energy_type = "Electric" if voted_color == 'green' else "Normal"
            record['energy_type'] = energy_type
            
            # Step 2. 微观时空轨迹结算
            if 'trajectory' in record:
                self._calculate_and_save_history(tid, record, final_type_str)

            # Step 3. 宏观数据入库
            # 从 record 中安全获取刚算好的工况，如果因为轨迹太短没算出来就给兜底值
            dominant_opmodes = record.get('dominant_opmodes', ["Unknown"])
            
            # 提取基础车型用于存入数据库 (比如 "LDV-Gasoline" -> "LDV")
            base_vehicle_type = final_type_str.split('-')[0] if final_type_str else "LDV"

            # 根据正常结算/临时结算状态写入状态判断
            settlement_status = "Unsettled" if is_continued else "Settled"

            if getattr(self, 'current_session_id', None):
                # 1. 宏观统计入库 (Veh_Sum - 供前端预览)
                self.db.insert_veh_sum(
                    session_id=self.current_session_id,
                    tid=tid, 
                    record=record, 
                    vehicle_type=base_vehicle_type, 
                    energy_type=energy_type, 
                    dominant_opmodes=dominant_opmodes,
                    settlement_status=settlement_status
                )

                # 2. 微观轨迹入库 (Veh_Raw - 供后端分析)
                if 'trajectory_blob_data' in record:
                    self.db.insert_veh_raw(
                        session_id=self.current_session_id,
                        tid=tid,
                        vehicle_type=base_vehicle_type,
                        energy_type=energy_type,
                        entry_time=record.get('first_time', 0.0),
                        exit_time=record.get('last_seen_time', 0.0),
                        trajectory=record['trajectory_blob_data']
                    )
    
    def _calculate_and_save_history(self, tid, record, final_type_str):
        trajectory = record.get('trajectory', [])
        if len(trajectory) < 3: return 

        raw_x = np.array([p.get('raw_x', 0.0) for p in trajectory])
        raw_y = np.array([p.get('raw_y', 0.0) for p in trajectory])
        timestamps = np.array([p.get('timestamp', 0.0) for p in trajectory])

        # 直接调用领域服务，一行代码完成所有降维、平滑与求导逻辑 (运动学职责)
        sm_x, sm_y, speeds, accels = self.smoother.process_1d(timestamps, raw_x, raw_y)

        pts_phys_clean = [] # 用于收集平滑过滤后的物理坐标
        vsp_data_for_mapper = []  # 用于收集 OpModeMapper 需要的时序数据
        
        # 覆盖原始轨迹并计算 VSP
        for i in range(len(trajectory)):
            # 1. 基础物理量赋值
            trajectory[i]['raw_x'] = float(sm_x[i])
            trajectory[i]['raw_y'] = float(sm_y[i])
            trajectory[i]['speed'] = float(speeds[i])
            trajectory[i]['accel'] = float(accels[i])
            
            # 2. 计算 VSP
            vsp_val = self.vsp_calc.calculate(
                v_ms=float(speeds[i]),
                a_ms2=float(accels[i]),
                vehicle_type=final_type_str.split('-')[0] 
            )
            trajectory[i]['vsp'] = vsp_val # 顺便存入轨迹字典，方便UI预览或除错

            # 3. 收集坐标
            pts_phys_clean.append([float(sm_x[i]), float(sm_y[i])])

            # 4. 按 opmode_mapper 需要的格式收集时序数据
            vsp_data_for_mapper.append({
                'timestamp': float(timestamps[i]),
                'v_ms': float(speeds[i]),
                'vsp': vsp_val
            })

        record['trajectory'] = trajectory

        # 批处理整条平滑轨迹，提取主导工况，并将结果直接挂载到 record 字典上
        record['dominant_opmodes'] = self.opmode_mapper.extract_dominant_opmodes(vsp_data_for_mapper)

        # 将平滑后的纯净坐标交给 SpatialAnalyzer，计算纯几何距离 (空间职责)
        record['total_distance_m'] = self.spatial.calculate_geometric_distance(pts_phys_clean)

        self.latest_exit_record = {
            'tid': tid, 'record': record, 'type_str': final_type_str
        }

        # 微观轨迹入库准备 (准备喂给 Veh_Raw 表的 BLOB)
        # 数据库写入的目标频率 (5Hz)
        db_fps = 5.0
        target_db_dt = 1.0 / db_fps
        
        # 复用平滑器中带尾点安全间距处理的降采样逻辑
        valid_indices = KinematicsSmoother.get_downsampled_indices(timestamps, target_db_dt)
        
        # 构建干净的物理时空轨迹数据结构，隔离底层的CV像素与帧率参数
        trajectory_for_db = []
        # 获取当前切片真实的起点时间
        first_time = float(record.get('first_time', 0.0))
        for i in valid_indices:
            p = trajectory[i]
            # 利用 first_time 严格剔除为了平滑算法而保留的历史重叠点
            if float(p['timestamp']) >= first_time:
                trajectory_for_db.append({
                    'timestamp': float(p['timestamp']),
                    'x': float(p['raw_x']),   # 映射为纯粹的物理横向坐标 (米)
                    'y': float(p['raw_y']),   # 映射为纯粹的物理纵向坐标 (米)
                    'v': float(p['speed']),   # 平滑后的物理速度 (m/s)
                    'a': float(p['accel']),   # 平滑后的物理加速度 (m/s^2)
                    'vsp': float(p.get('vsp', 0.0)) # 单点比功率
                })

        # 将打包好的纯净轨迹挂载到 record，供 _handle_exits 写入 Veh_Raw
        record['trajectory_blob_data'] = trajectory_for_db

    def _prepare_labels(self, detections, frame_shape):
        img_h, img_w = frame_shape[:2]  
        labels = []
        for i, (tid, raw_class_id) in enumerate(zip(detections.tracker_id, detections.class_id)):
            record = self.registry.get_record(tid)
            if not record:
                continue 
                
            voted_class_id = record['class_id']
            data = LabelData(track_id=tid, class_id=voted_class_id)
            
            # 直接查缓存判定
            plate_info = self.plate_cache.get(tid)
            
            if plate_info:
                current_color = plate_info['color']
                _, final_type = self.classifier.resolve_type(
                    voted_class_id, 
                    plate_color_override=current_color
                )
                data.display_type = final_type
                data.plate_color = current_color
                
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                
                # 与裁切时的 Padding 规则保持一致
                bw = x2 - x1
                bh = y2 - y1
                pad_x = int(bw * 0.05)
                pad_top = int(bh * 0.05)
                pad_bottom = int(bh * 0.20)
                
                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_top)
                crop_x2 = min(img_w, x2 + pad_x)
                crop_y2 = min(img_h, y2 + pad_bottom)
                
                crop_w = crop_x2 - crop_x1
                crop_h = crop_y2 - crop_y1
                
                rel_lms = plate_info['rel_landmarks']
                # 将历史比例映射到当前车辆的位置
                abs_lms = rel_lms * np.array([crop_w, crop_h]) + np.array([crop_x1, crop_y1])
                data.plate_points = abs_lms

                # ====== 调试打印 ======
                logger.debug(f"[DEBUG] 成功生成 TID={tid} 的 UI 车牌框坐标")
            else:
                # 即使没找到车牌，也要用分类器的基础兜底逻辑解析车型 (例如 "LDV")
                _, final_type = self.classifier.resolve_type(voted_class_id, plate_color_override=None)
                data.display_type = final_type
            
            labels.append(data)
            
        return labels

    def cleanup(self, final_frame_id):
        print("\n[Engine] 正在清理资源...")
        self.camera.stop()  # 停止 GStreamer 管道

        # 停止热成像后台线程
        if self.thermal_cam:
            self.thermal_cam.stop()

        # 停止 NTP 时钟同步器的后台线程
        if hasattr(self, 'time_sync'):
            self.time_sync.stop()
            print("[Engine] 时钟同步守护线程已停止。")
        
        print("[Engine] 保存剩余车辆数据...")
        # 伪造一个未来的绝对时间戳 (当前时间 + 1000秒)，强制所有未结算的车辆触发超时离场
        import time
        force_exit_timestamp = time.time() + 1000.0
        self._handle_exits(final_frame_id + 1000, force_exit_timestamp)

        if getattr(self, 'plate_worker', None):
            print("[Engine] 正在强制回收 OCR 子进程...")
            self.plate_worker.stop()

        # 宣告采集任务结束
        if getattr(self, 'current_session_id', None):
            self.db.complete_session(self.current_session_id, time.time())

        self.db.close()

    def _print_profile_stats(self):
        """打印每 30 帧的性能监控报表 (已接入 logging 模块并支持开关)"""
        # 1. 检查配置：是否开启性能探针 (默认关闭)
        if not getattr(self.cfg, 'ENABLE_PROFILE_LOG', False):
            # 即使不输出，也要重置字典，防止长期运行导致内存累积
            self.profile_stats.clear()
            self.profile_frames = 0
            return

        # 2. 拼接多行字符串：必须一次性合成一个大字符串再传给 logger
        # 防止在多线程/多进程高并发时，性能日志被其他普通日志从中间横向切断
        log_lines = ["", "="*45, "[性能探针] 过去 30 帧平均耗时 (ms/帧)", "-"*45]
        
        keys = sorted(self.profile_stats.keys())
        for k in keys:
            avg_ms = (self.profile_stats[k] / self.profile_frames) * 1000
            if k == '00_total_loop':
                log_lines.append(f"{'-'*45}\n> {k.ljust(23)}: {avg_ms:6.2f} ms")
            else:
                log_lines.append(f"  {k.ljust(23)}: {avg_ms:6.2f} ms")
        
        log_lines.append("="*45)

        # 3. 通过标准日志模块输出 (使用 DEBUG 级别，保持终端整洁)
        logger.debug("\n".join(log_lines))

        # 4. 重置统计
        self.profile_stats.clear() # clear() 会清空 defaultdict，比遍历赋值更高效
        self.profile_frames = 0
