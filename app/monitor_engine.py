import cv2
import numpy as np
import supervision as sv
import time
from collections import defaultdict
from ui.renderer import resize_with_pad, LabelData
from infra.time.ntp_sync import TimeSynchronizer
from domain.physics.spatial_analyzer import SpatialAnalyzer
from domain.physics.kinematics_smoother import KinematicsSmoother
from perception.vision_pipeline import VisionPipeline

class TrafficMonitorEngine:
    """
    [应用层] 交通监测主引擎 (Traffic Monitor Engine) - Edge Version (Hailo-8 架构)

    该类作为系统的核心控制器，接收 GStreamer 管道输出的视频帧与硬件推理元数据，
    负责协调透视变换、车牌分类、物理估算、排放计算及数据存储等各个子模块。
    """

    def __init__(self, config, components, frame_callback=None):
        self.cfg = config
        self.comps = components
        self.frame_callback = frame_callback  # 保存回调函数
        self._is_running = True # 增加运行状态标志位，用于安全退出
        self.time_sync = TimeSynchronizer() # 初始化时钟同步器
        
        # --- 核心组件引用 ---
        # 注意：model (YOLO) 和 tracker (ByteTrack) 已被移除，交由硬件管道完成
        self.camera = components['camera']          # GstPipelineManager (GStreamer管道)
        self.registry = components['registry']      # 车辆注册表 (内存数据库)
        self.visualizer = components['visualizer']  # 可视化渲染器
        self.db = components['db']                  # 持久化存储 (SQLite)

        # 替换掉了原来的 plate_classifier
        self.plate_worker = components.get('plate_worker') 
        
        # --- 状态缓存 ---
        self.plate_cache = {} 
        self.plate_retry = {} 
        
        # --- 功能开关 ---
        self.debug_mode = config.DEBUG_MODE
        self.motion_on = config.ENABLE_MOTION       
        self.ocr_on = config.ENABLE_OCR             

        # --- 标签到 ID 的映射字典 (适配原有逻辑) ---
        self.label_map = {
            "car": self.cfg.YOLO_CLASS_CAR,
            "bus": self.cfg.YOLO_CLASS_BUS,
            "truck": self.cfg.YOLO_CLASS_TRUCK
        }
        
        # 修复潜在的 AttributeError: 恢复对基础车型分类器(用于解析车辆类型逻辑)的引用
        self.classifier = components.get('classifier')

        # 初始化视觉处理流水线
        self.vision = VisionPipeline(fps=config.FPS, label_map=self.label_map)

        # 初始化空间分析器和运动滤波器
        self.spatial = SpatialAnalyzer()
        self.smoother = KinematicsSmoother(max_window=15)

        # 从组件字典中获取热成像实例
        self.thermal_cam = components.get('thermal_cam')

    def run(self):
        """
        启动基于 GStreamer 轮询的主处理循环。
        """
        print(f">>> [Engine] 正在启动硬件加速管道...")
        self.camera.start()

        # 启动热成像后台采集线程
        if self.thermal_cam:
            print(">>> [Engine] 检测到热成像模块，正在启动采集任务...")
            self.thermal_cam.start()
        
        # 初始化录制相关的变量
        video_info = None
        sink = None
        frame_id = 0

        print(f">>> [Engine] 等待视频流接入...")
        
        try:
            # 初始化 FPS 计算变量
            prev_time = time.time()
            frame_count = 0
            current_fps = 0.0

            # 软件级限速计算器
            target_delay = 1.0 / self.cfg.FPS

            while True:
                # 记录循环开始时间
                loop_start = time.time()

                # 1. 阻塞拉取底层已经处理好的数据
                frame, buffer = self.camera.read()

                # 拿到帧的同时打上时间戳
                frame_timestamp = self.time_sync.get_precise_timestamp()
                
                if frame is None or buffer is None:
                    # 如果流未就绪，稍微休眠防止 CPU 空转
                    time.sleep(0.01)
                    continue
                
                frame_id += 1

                # 计算每秒的实时帧率 (平滑显示，避免数值疯狂闪烁)
                frame_count += 1
                now = time.time()
                if now - prev_time >= 1.0:
                    current_fps = frame_count / (now - prev_time)
                    prev_time = now
                    frame_count = 0
                
                # 2. 延迟初始化 VideoSink (因为需要确切知道输出的分辨率)
                if video_info is None:
                    h, w = frame.shape[:2]
                    video_info = sv.VideoInfo(width=w, height=h, fps=self.cfg.FPS)
                    sink = sv.VideoSink(self.cfg.TARGET_VIDEO_PATH, video_info=video_info)
                    sink.__enter__()
                    print(f">>> [Engine] 视频流已接入: {w}x{h} @ {self.cfg.FPS}fps")
                
                    #  第一帧到达时，动态适配底层坐标系
                    if 'norm_source_points' in self.comps:
                        # 拿到 0~1 的归一化点，乘以底层管道输出的真实宽高
                        pts = self.comps['norm_source_points'].copy()
                        pts[:, 0] *= w
                        pts[:, 1] *= h
                        
                        # 把算好的绝对坐标塞回绘图器
                        self.visualizer.calibration_points = pts.astype(np.int32)
                        
                        # 重新初始化视角转换器，确保物理速度/位移计算正确
                        from perception.math.geometry import ViewTransformer
                        transformer = ViewTransformer(pts, self.comps['target_points'])
                        self.comps['transformer'] = transformer

                        # 给空间分析器注入映射能力
                        self.spatial.set_transformer(transformer)

                # --- 核心处理流水线 ---
                annotated_frame = self.process_frame(frame, buffer, frame_id, current_fps, frame_timestamp)
                
                # --- 写入结果视频 ---
                if sink:
                    sink.write_frame(annotated_frame)
                
                # --- 实时预览 ---
                if self.frame_callback:
                    # 为了性能，直接在 Engine 端缩放到树莓派屏幕尺寸 800x480
                    display = resize_with_pad(annotated_frame, (800, 480))
                    self.frame_callback(display)
                
                if not getattr(self, '_is_running', True):
                    break

                # 软件限速：如果处理得比 30FPS 快，就等一会儿，防止画面快进！
                elapsed = time.time() - loop_start
                if elapsed < target_delay:
                    time.sleep(target_delay - elapsed)
                    
        except KeyboardInterrupt:
            print("\n>>> [Engine] 接收到退出信号...")
        finally:
            if sink:
                sink.__exit__(None, None, None)
            self.cleanup(frame_id)

    def process_frame(self, frame, buffer, frame_id, current_fps=0.0, frame_timestamp=0.0):
        """
        单帧处理流水线 (混合架构版)。
        """
        h, w = frame.shape[:2]
        
        # --- Step 1: 视觉感知与目标追踪 (交由感知层流水线处理) ---
        detections = self.vision.process(frame, buffer)

        # --- Step 2: 注册表更新 (Registry Update) ---
        self.registry.update(detections, frame_id, frame_timestamp, None)
        self._handle_exits(frame_id, frame_timestamp)
        
        # --- Step 3: 异步车牌分类 ---
        if self.ocr_on and self.plate_worker:
            # 1. 投递新任务
            self._dispatch_plate_tasks(frame, frame_id, detections)
            # 2. 收割已完成的结果 (不阻塞)
            self._collect_plate_results()

        # --- Step 4: 物理轨迹打点与动态死区判定 ---
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
                        fallback_dt = 1.0 / self.cfg.FPS
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

        # --- Step 5: 可视化渲染 ---
        label_data_list = self._prepare_labels(detections)
        return self.visualizer.render(frame, detections, label_data_list, fps=current_fps)

    def _dispatch_plate_tasks(self, frame, frame_id, detections):
        """派发任务给子进程：将车身裁剪出来，非阻塞地放入队列"""
        img_h, img_w = frame.shape[:2]
        if frame_id % self.cfg.OCR_INTERVAL != 0:
            return

        # 获取用户标定的 ROI 4 个角点
        pts = self.visualizer.calibration_points
        if pts is None or len(pts) < 4:
            return
            
        # 提取 ROI 的“下边线” (左下角 BL 与 右下角 BR)
        p1 = pts[0].astype(np.float32)
        p2 = pts[1].astype(np.float32)
        
        # 构建底边向量
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-4: return

        for tid, box in zip(detections.tracker_id, detections.xyxy):
            # 冷却检查
            if frame_id - self.plate_retry.get(tid, -999) < self.cfg.OCR_RETRY_COOLDOWN:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            
            # 使用检测框“底边中点”作为车辆物理触地点
            bc_x, bc_y = (x1 + x2) / 2.0, float(y2)
            pt_vec = np.array([bc_x, bc_y]) - p1

            # 点到直线的垂直像素距离 (叉乘 / 底边长)
            dist_to_bottom = np.abs(np.cross(line_vec, pt_vec)) / line_len
            
            # 车辆横向投影比例 
            # (0 代表车辆位于左下角，1 代表位于右下角，超出这个范围说明车在线段延长线外)
            proj_ratio = np.dot(pt_vec, line_vec) / (line_len ** 2)

            # 智能触发条件
            # 条件 A: 距离 ROI 下边线小于画面高度的 25% (如 1080p 下约 270 像素宽度的触发带)
            # 条件 B: 车辆横向在底边之间 (放宽 ±10% 的容差，允许压线)
            if dist_to_bottom < (img_h * 0.25) and -0.1 < proj_ratio < 1.1:
                
                # 尺寸筛选
                scaled_min_area = self.cfg.MIN_PLATE_AREA * (img_w / 1920.0) * (img_h / 1080.0)
                if (x2-x1)*(y2-y1) > scaled_min_area:
                    vehicle_crop = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)].copy()
                    if vehicle_crop.size > 0:
                        if self.plate_worker.push_task(tid, vehicle_crop):
                            self.plate_retry[tid] = frame_id

    def _collect_plate_results(self):
        """非阻塞地从子进程收取计算结果并入库"""
        results = self.plate_worker.get_results()
        
        color_thresholds = {
            'green': 0.60,  
            'blue': 0.85,   
            'yellow': 0.75  
        }

        # 只要这辆车历史上有 N 帧被识别为绿/黄，就强制锁定
        VETO_THRESHOLD = 1

        for tid, color_type, conf, rel_landmarks in results:
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

                # 3. 更新缓存，允许更新坐标(rel_landmarks)，但颜色可能被强行纠正为绿/黄
                self.plate_cache[tid] = {
                    'color': final_color_for_ui,
                    'rel_landmarks': rel_landmarks
                }

    def _handle_exits(self, frame_id, current_timestamp):
        """
        处理离场车辆：执行最终结算、生成报表并入库。
        """
        for tid, record in self.registry.check_exits(frame_id, current_timestamp):
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
                    # 如果没有达到少数派阈值，才退回到普通的置信度投票 (大概率会投出 blue)
                    scores = defaultdict(float)
                    for entry in history:
                        scores[entry['color']] += entry.get('conf', 1.0)
                    voted_color = max(scores, key=scores.get)
            
            record['final_plate_color'] = voted_color
            
            # Step 2. 微观时空轨迹与 VSP 结算 (核心逻辑)
            if 'trajectory' in record:
                self._calculate_and_save_history(tid, record, final_type_str)

            # Step 3. 宏观数据入库
            self.db.insert_macro(tid, record, final_type_str, final_plate)
    
    def _calculate_and_save_history(self, tid, record, final_type_str):
        trajectory = record.get('trajectory', [])
        if len(trajectory) < 3: return 

        raw_x = np.array([p.get('raw_x', 0.0) for p in trajectory])
        raw_y = np.array([p.get('raw_y', 0.0) for p in trajectory])
        timestamps = np.array([p.get('timestamp', 0.0) for p in trajectory])

        # 直接调用领域服务，一行代码完成所有降维、平滑与求导逻辑 (运动学职责)
        sm_x, sm_y, speeds, accels = self.smoother.process_1d(timestamps, raw_x, raw_y)

        # 增加一个列表，用于收集平滑过滤后的“纯净物理坐标”
        pts_phys_clean = []
        
        # 覆盖原始轨迹并暴露给 UI Dashboard 展示
        for i in range(len(trajectory)):
            trajectory[i]['raw_x'] = float(sm_x[i])
            trajectory[i]['raw_y'] = float(sm_y[i])
            trajectory[i]['speed'] = float(speeds[i])
            trajectory[i]['accel'] = float(accels[i])
            
            # 收集坐标
            pts_phys_clean.append([float(sm_x[i]), float(sm_y[i])])

        record['trajectory'] = trajectory 

        # 将平滑后的纯净坐标交给 SpatialAnalyzer，计算纯几何距离 (空间职责)
        record['total_distance_m'] = self.spatial.calculate_geometric_distance(pts_phys_clean)

        self.latest_exit_record = {
            'tid': tid, 'record': record, 'type_str': final_type_str
        }

        # 基于物理时间的动态抽样 (与平滑器里的逻辑对齐)
        db_fps = 5.0
        target_db_dt = 1.0 / db_fps
        trajectory_for_db = []
        
        if trajectory:
            trajectory_for_db.append(trajectory[0]) # 始终保留起点
            last_t = trajectory[0].get('timestamp', 0.0)
            
            for point in trajectory[1:]:
                curr_t = point.get('timestamp', 0.0)
                if curr_t - last_t >= target_db_dt:
                    trajectory_for_db.append(point)
                    last_t = curr_t
                    
            # 始终保留终点，防止离场时刻缺失
            if trajectory[-1] not in trajectory_for_db:
                trajectory_for_db.append(trajectory[-1])

        # 极简且降频的微观轨迹入库
        for point in trajectory_for_db:
            db_payload = {
                'timestamp': point.get('timestamp', 0.0),
                'ipm_x': point.get('raw_x', 0.0),   
                'ipm_y': point.get('raw_y', 0.0)    
            }
            self.db.insert_micro(point.get('frame_id', 0), tid, db_payload)

        self.db.flush_micro_buffer()

    def _handle_ocr(self, frame, frame_id, detections):
        """
        处理车牌识别任务 (OCR)。
        仅对位于 ROI 中心区域、且图像质量合格的车辆触发。
        """
        worker = self.comps.get('ocr_worker')
        if not worker: return

        img_h, img_w = frame.shape[:2]
        
        # Step 1. 任务分发 (按间隔触发)
        if frame_id % self.cfg.OCR_INTERVAL == 0:
            for tid, box, cid in zip(detections.tracker_id, detections.xyxy, detections.class_id):
                # --- 冷却检查 ---
                if frame_id - self.plate_retry.get(tid, -999) < self.cfg.OCR_RETRY_COOLDOWN:
                    continue
                
                # --- 几何筛选: 仅处理画面中心区域的车辆 ---
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1+x2)/2, (y1+y2)/2
                if not (0.1*img_w < cx < 0.9*img_w and 0.4*img_h < cy < 0.98*img_h):
                    continue
                
                # --- 尺寸筛选 ---
                if (x2-x1)*(y2-y1) > self.cfg.MIN_PLATE_AREA:
                    crop = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)].copy()
                    if worker.push_task(tid, crop, cid):
                        self.plate_retry[tid] = frame_id

        # Step 2. 结果回收
        for (tid, color, conf, area) in worker.get_results():
            self.registry.add_plate_history(tid, color, area, conf)
            if conf > self.cfg.OCR_CONF_THRESHOLD:
                self.plate_cache[tid] = color
                if tid in self.plate_retry: del self.plate_retry[tid]

    def _prepare_labels(self, detections, landmarks_dict=None):
        """
        [修改版 UI 数据适配] 专为极简显示设计 + 动态车牌锚定
        """
        labels = []
        # 注意这里加了 enumerate(zip(...)) 以便获取当前车辆的索引 i
        for i, (tid, raw_class_id) in enumerate(zip(detections.tracker_id, detections.class_id)):
            record = self.registry.get_record(tid)
            voted_class_id = record['class_id'] if record else int(raw_class_id)

            data = LabelData(track_id=tid, class_id=voted_class_id)
            
            # --- 1. 强制设定极简英文分类 ---
            if voted_class_id == self.cfg.YOLO_CLASS_CAR: data.display_type = "car"
            elif voted_class_id == self.cfg.YOLO_CLASS_BUS: data.display_type = "bus"
            elif voted_class_id == self.cfg.YOLO_CLASS_TRUCK: data.display_type = "truck"
            else: data.display_type = "vehicle"
                
            # --- 2. 动态计算车牌框 (相对坐标映射核心逻辑) ---
            plate_info = self.plate_cache.get(tid)
            if plate_info:
                data.plate_color = plate_info['color']
                
                # 获取该车在【当前这刚刚到来的一帧】里的最新检测框
                x1, y1, x2, y2 = detections.xyxy[i]
                vw = x2 - x1
                vh = y2 - y1
                
                # 将 0~1 的相对坐标放大到当前车身尺寸，并加上车身的左上角偏移
                rel_lms = plate_info['rel_landmarks']
                abs_lms = rel_lms * np.array([vw, vh]) + np.array([x1, y1])
                data.plate_points = abs_lms
            
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
        self._handle_exits(final_frame_id + 1000)
        self.db.close()
