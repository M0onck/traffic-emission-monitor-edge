import cv2
import numpy as np
import supervision as sv
import time
from collections import defaultdict
from ui.renderer import resize_with_pad, LabelData

# 引入 Hailo 元数据解析库 (需要在运行环境/设备端安装有 tappas 的 python 绑定)
try:
    import hailo
except ImportError:
    print(">>> [Warn] 找不到 hailo 模块，请确保在树莓派/Hailo环境下运行，或安装了 Tappas。")

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
        self.emission_req = config.ENABLE_EMISSION  

        # --- 标签到 ID 的映射字典 (适配原有逻辑) ---
        self.label_map = {
            "car": self.cfg.YOLO_CLASS_CAR,
            "bus": self.cfg.YOLO_CLASS_BUS,
            "truck": self.cfg.YOLO_CLASS_TRUCK
        }

        # 分类器引用更新：原先完整的 OCR 已被替换为仅针对车牌属性的小模型分类器
        self.plate_classifier = components.get('plate_classifier') 
        
        # 修复潜在的 AttributeError: 恢复对基础车型分类器(用于解析车辆类型逻辑)的引用
        self.classifier = components.get('classifier')

    def run(self):
        """
        启动基于 GStreamer 轮询的主处理循环。
        """
        print(f">>> [Engine] 正在启动硬件加速管道...")
        self.camera.start()
        
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

            # 🚨 软件级限速计算器
            target_delay = 1.0 / self.cfg.FPS

            while True:
                # 记录循环开始时间
                loop_start = time.time()

                # 1. 阻塞拉取底层已经处理好的数据
                frame, buffer = self.camera.read()
                
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
                        self.comps['transformer'] = ViewTransformer(pts, self.comps['target_points'])

                # --- 核心处理流水线 ---
                annotated_frame = self.process_frame(frame, buffer, frame_id, current_fps)
                
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

                # 🚨 终极软件限速：如果处理得比 30FPS 快，就等一会儿，防止画面快进！
                elapsed = time.time() - loop_start
                if elapsed < target_delay:
                    time.sleep(target_delay - elapsed)
                    
        except KeyboardInterrupt:
            print("\n>>> [Engine] 接收到退出信号...")
        finally:
            if sink:
                sink.__exit__(None, None, None)
            self.cleanup(frame_id)

    def process_frame(self, frame, buffer, frame_id, current_fps=0.0):
        """
        单帧处理流水线 (混合架构版)。
        """
        h, w = frame.shape[:2]
        
        # --- Step 1: 解析 Hailo Metadata (从底层接收推理结果) ---
        xyxy, class_ids, confs, tracker_ids = [], [], [], []
        landmarks_dict = {} # {track_id: np.array([[x1,y1], [x2,y2]...])}

        try:
            roi = hailo.get_roi_from_buffer(buffer)
            hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
            
            for det in hailo_detections:
                # 获取检测框
                bbox = det.get_bbox()
                x1, y1 = int(bbox.xmin() * w), int(bbox.ymin() * h)
                x2, y2 = int(bbox.xmax() * w), int(bbox.ymax() * h)
                
                # 获取 Tracking ID (由底层的 hailotracker 挂载)
                track_id = -1
                for obj in det.get_objects_typed(hailo.HAILO_UNIQUE_ID):
                    track_id = obj.get_id()
                    break
                
                if track_id == -1: continue # 跳过没有追踪ID的目标
                
                # 获取标签，并进行严格的白名单过滤
                label = det.get_label()
                if label not in self.label_map:
                    continue  # 直接丢弃所有非 car/bus/truck 的目标（如 person, bicycle 等）
                    
                cid = self.label_map[label]
                
                # 获取关键点 (在 C++ .so 中挂载的)
                lm_pts = []
                for lm_obj in det.get_objects_typed(hailo.HAILO_LANDMARKS):
                    for pt in lm_obj.get_points():
                        lm_pts.append([pt.x() * w, pt.y() * h])
                
                xyxy.append([x1, y1, x2, y2])
                class_ids.append(cid)
                confs.append(det.get_confidence())
                tracker_ids.append(track_id)
                if len(lm_pts) == 4: # 假设我们保存了车牌的4个角点
                    landmarks_dict[track_id] = np.array(lm_pts, dtype=np.float32)

        except Exception as e:
            # 如果解析出错（或者在非真实设备环境运行），构建空的数据防止崩溃
            pass
            
        # 构建对下游完全透明的 supervision 结构格式
        if len(xyxy) > 0:
            detections = sv.Detections(
                xyxy=np.array(xyxy, dtype=np.float32),
                confidence=np.array(confs, dtype=np.float32),
                class_id=np.array(class_ids, dtype=int),
                tracker_id=np.array(tracker_ids, dtype=int)
            )
            detections = detections.with_nms(threshold=0.6, class_agnostic=True)
        else:
            detections = sv.Detections.empty()
            detections.tracker_id = np.array([], dtype=int)
            detections.class_id = np.array([], dtype=int)
            detections.confidence = np.array([], dtype=np.float32)

        # --- Step 2: 注册表更新 (Registry Update) ---
        # 移除了 self.model 参数，因为车辆图片特征提取已交由前端模型完成
        self.registry.update(detections, frame_id, None)
        self._handle_exits(frame_id)
        
        # --- Step 3: 异步车牌分类 ---
        if self.ocr_on and self.plate_worker:
            # 1. 投递新任务
            self._dispatch_plate_tasks(frame, frame_id, detections)
            # 2. 收割已完成的结果 (不阻塞)
            self._collect_plate_results()

        # --- Step 4: 物理参数估算 (Kinematics) ---
        # (这部分代码与原版保持完全一致，因为它依赖于 Detections 格式)
        kinematics_data = {}
        realtime_opmodes = {}
        
        if self.motion_on and self.comps.get('kinematics'):
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed = self.comps['transformer'].transform_points(points)
            roi_bounds = self.comps['transformer'].get_roi_vertical_bounds()
            
            kinematics_data = self.comps['kinematics'].update(
                detections, transformed, frame.shape, roi_y_range=roi_bounds
            )
            
            vsp_calc = self.comps.get('vsp_calculator')
            brake_model = self.comps.get('brake_model')
            opmode_calc = getattr(brake_model, 'opmode_calculator', None) if brake_model else None
            
            if vsp_calc and opmode_calc:
                for tid, k_data in kinematics_data.items():
                    mask = detections.tracker_id == tid
                    if not np.any(mask): continue
                    class_id = int(detections.class_id[mask][0])
                    v, a = k_data['speed'], k_data['accel']
                    vsp = vsp_calc.calculate(v, a, class_id)
                    realtime_opmodes[tid] = opmode_calc.get_opmode(v, a, vsp)

            # 记录原始轨迹点
            tid_to_pixel = {tid: pt for tid, pt in zip(detections.tracker_id, points)}
            transformer = self.comps['transformer']

            for tid, k_data in kinematics_data.items():
                raw_point = tid_to_pixel.get(tid)
                if raw_point is not None and transformer.is_in_roi(raw_point):
                    self.registry.append_kinematics(
                        tid, frame_id, 
                        k_data['speed'], k_data['accel'],
                        raw_x=k_data['curr_x'], raw_y=k_data['curr_y'],
                        pixel_x=raw_point[0], pixel_y=raw_point[1]
                    )

        # --- Step 5: 排放模型 ---
        emission_data = {}

        # --- Step 6: 可视化渲染 ---
        label_data_list = self._prepare_labels(
            detections,
            landmarks_dict
        )
        return self.visualizer.render(frame, detections, label_data_list, fps=current_fps)

    def _dispatch_plate_tasks(self, frame, frame_id, detections):
        """派发任务给子进程：将车身裁剪出来，非阻塞地放入队列"""
        img_h, img_w = frame.shape[:2]
        if frame_id % self.cfg.OCR_INTERVAL != 0:
            return

        for tid, box in zip(detections.tracker_id, detections.xyxy):
            # 冷却检查
            if frame_id - self.plate_retry.get(tid, -999) < self.cfg.OCR_RETRY_COOLDOWN:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1+x2)/2, (y1+y2)/2

            # 🚨 放宽判定区域，让车刚进镜头就开始被截取，留给后台充足的计算时间！
            if not (0.1*img_w < cx < 0.9*img_w and 0.2*img_h < cy < 0.95*img_h):
                continue
                
            # 动态缩放面积阈值 (假设已适配720p或1080p)
            scaled_min_area = self.cfg.MIN_PLATE_AREA * (img_w / 1920.0) * (img_h / 1080.0)
            if (x2-x1)*(y2-y1) > scaled_min_area:
                vehicle_crop = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)].copy()
                if vehicle_crop.size > 0:
                    # 尝试将裁剪后的图像推入队列
                    if self.plate_worker.push_task(tid, vehicle_crop):
                        self.plate_retry[tid] = frame_id # 只有成功推入才刷新冷却时间

    def _collect_plate_results(self):
        """非阻塞地从子进程收取计算结果并入库"""
        results = self.plate_worker.get_results()
        for tid, color_type, conf in results:
            # 🚀 探针 3：查看主进程收到的数据和配置的阈值冲突
            print(f"[Engine Probe] <- 收到后台结果 ID: {tid}, 颜色: {color_type}, Conf: {conf:.3f}, 配置文件阈值: {self.cfg.OCR_CONF_THRESHOLD}", flush=True)
            
            # 暂时将阈值强行设为极低，确保数据能流进去
            if conf > -999.0: 
                self.registry.add_plate_history(tid, color_type, 1.0, conf)
                self.plate_cache[tid] = color_type
            else:
                print(f"[Engine Probe] X 结果被拦截！", flush=True)

    def _handle_exits(self, frame_id):
        """
        处理离场车辆：执行最终结算、生成报表并入库。
        """
        for tid, record in self.registry.check_exits(frame_id):
            # Step 1. 解析最终属性 (车牌、细分车型)
            final_plate, final_type_str = self.classifier.resolve_type(
                record['class_id'], record.get('plate_history', [])
            )
            
            # Step 2. 几何距离重算
            self._recalculate_distance_geometric(record)

            # Step 3. 微观排放结算 (核心逻辑)
            if self.emission_req and 'trajectory' in record:
                self._calculate_and_save_history(tid, record, final_type_str)

            # Step 4. 宏观数据入库
            self.db.insert_macro(tid, record, final_type_str, final_plate)

            # Step 5. 控制台报告 (Debug Mode)
            if self.debug_mode and self.comps.get('reporter'):
                self.comps['reporter'].print_exit_report(
                    tid, record, self.comps.get('kinematics'), self.classifier
                )
    
    def _recalculate_distance_geometric(self, record):
        """
        [算法优化] 基于几何路径重算总里程。
        
        使用原始像素点的透视变换结果累加欧氏距离。
        相比于实时速度积分 (\int_{0}^{t} v(t) \, \mathrm{dt})，该方法不受滤波滞后和起步噪声影响，
        能更真实地反映车辆的物理位移。
        """
        trajectory = record.get('trajectory', [])
        if len(trajectory) < 2: return

        # Step 1. 提取有效像素点
        pixels = []
        for p in trajectory:
            if p.get('pixel_x') is not None and p.get('pixel_y') is not None:
                pixels.append([p['pixel_x'], p['pixel_y']])
        
        if len(pixels) < 2: return
        
        # Step 2. 批量透视变换 (Pixel -> Meter)
        pts_phys = self.comps['transformer'].transform_points(np.array(pixels))
        
        # Step 3. 累加线段长度
        diffs = pts_phys[1:] - pts_phys[:-1]
        dists = np.linalg.norm(diffs, axis=1)
        record['total_distance_m'] = float(np.sum(dists))

    def _calculate_and_save_history(self, tid, record, final_type_str):
        """
        [核心逻辑] 离场后微观数据结算 (Post-departure Settlement)

        流程：
        1. 轨迹清洗 (Trimming): 去除进出画面边缘的不稳定数据。
        2. 全局重构 (Global Refinement): 后处理运动轨迹，生成符合物理规律的速度/加速度曲线。
        3. 参数计算 (Physics): 计算 VSP 和 原始 OpMode。
        4. 序列清洗 (Sequence Cleaning): 使用状态机消除非法的工况跳变。
        5. 排放计算 (Emission): 查表计算最终排放量并入库。
        """
        trajectory = record.get('trajectory', [])

        # Step 1. 轨迹头尾清洗 (Trimming)
        TRIM_SIZE = 5 
        if len(trajectory) > (TRIM_SIZE * 2 + 5):
            trajectory = trajectory[TRIM_SIZE : -TRIM_SIZE]
            record['trajectory'] = trajectory # 回写以便 Reporter 使用
        else:
            return # 轨迹太短，放弃计算

        # Step 2. 全局轨迹重构 (Global Refinement)
        if len(trajectory) > 10 and 'raw_x' in trajectory[0]:
             trajectory = self._refine_trajectory_global(trajectory, record['class_id']) 

        # Step 3. 参数计算 (Physics)
        # --- 组件检查 ---
        vsp_calc = self.comps.get('vsp_calculator')
        brake_model = self.comps.get('brake_model')
        tire_model = self.comps.get('tire_model')
        opmode_calc = getattr(brake_model, 'opmode_calculator', None)
        
        if not (vsp_calc and brake_model and tire_model and opmode_calc):
            return

        # --- 准备计算参数 ---
        final_class_id = record['class_id']
        category = 'CAR'
        if final_class_id == self.cfg.YOLO_CLASS_BUS: category = 'BUS'
        elif final_class_id == self.cfg.YOLO_CLASS_TRUCK: category = 'TRUCK'
        is_electric = "electric" in final_type_str
        dt = 1.0 / self.cfg.FPS

        # --- 物理参数预计算 ---
        pre_calc_data = []
        raw_opmodes = []

        for point in trajectory:
            v = point['speed']
            a = point['accel']
            fid = point['frame_id']
            
            # 计算 VSP 和 原始 OpMode
            vsp = vsp_calc.calculate(v, a, final_class_id)
            raw_op = opmode_calc.get_opmode(v, a, vsp)
            
            pre_calc_data.append({'fid': fid, 'v': v, 'a': a, 'vsp': vsp})
            raw_opmodes.append(raw_op)

        # Step 4. OpMode 序列清洗
        clean_opmodes = self._clean_opmode_sequence(raw_opmodes)

        # Step 5. 排放结算与入库
        for i, data in enumerate(pre_calc_data):
            op_mode = clean_opmodes[i]
            v, a, vsp = data['v'], data['a'], data['vsp']
            
            # Step 5.1. 刹车排放 (含 EV 再生制动修正)
            brake_res = brake_model.calculate_single_point(
                v, a, vsp, final_class_id, dt, type_str=final_type_str
            )
            brake_emission = brake_res['emission_mass']

            # Step 5.2. 轮胎排放 (含 EV 重量惩罚)
            tire_base = tire_model._get_rate(category, op_mode)
            tire_factor = self.cfg.MASS_FACTOR_EV if is_electric else 1.0
            tire_emission = tire_base * tire_factor * dt

            # Step 5.3. 统计累加
            if hasattr(self.registry, 'accumulate_opmode'):
                self.registry.accumulate_opmode(record, op_mode)
                self.registry.accumulate_brake_emission(record, brake_emission)
                self.registry.accumulate_tire_emission(record, tire_emission)
            else:
                self.registry.update_emission_stats(record, op_mode, brake_emission)
                self.registry.update_tire_stats(record, tire_emission)

            # Step 5.4. 入库 (微观表)
            db_payload = {
                'type_str': final_type_str,
                'plate_color': "Resolved",
                'speed': v, 'accel': a, 'vsp': vsp,
                'op_mode': op_mode,
                'brake_emission': brake_emission,
                'tire_emission': tire_emission
            }
            self.db.insert_micro(data['fid'], tid, db_payload)
            
        # --- 强制刷写缓冲区 ---
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
        [UI 数据适配] 准备渲染标签。
        
        策略:
        1. 优先显示实时工况 (OpMode)，隐藏跳动的速度值。
        2. 整合排放模型输出的详细信息。
        """
        labels = []
        for tid, raw_class_id in zip(detections.tracker_id, detections.class_id):
            record = self.registry.get_record(tid)
            voted_class_id = record['class_id'] if record else int(raw_class_id)

            data = LabelData(track_id=tid, class_id=voted_class_id)
            
            # --- 1. 强制设定极简英文分类 ---
            if voted_class_id == self.cfg.YOLO_CLASS_CAR:
                data.display_type = "car"
            elif voted_class_id == self.cfg.YOLO_CLASS_BUS:
                data.display_type = "bus"
            elif voted_class_id == self.cfg.YOLO_CLASS_TRUCK:
                data.display_type = "truck"
            else:
                data.display_type = "vehicle"
                
            # --- 2. 挂载底层的车牌框点位与异步缓存的颜色 ---
            if landmarks_dict and tid in landmarks_dict:
                data.plate_points = landmarks_dict[tid]
            
            data.plate_color = self.plate_cache.get(tid) 
            
            labels.append(data)
            
        return labels

    def _refine_trajectory_global(self, trajectory, class_id):
        """
        [算法核心] 全局轨迹重构优化器

        本函数是数据后处理的核心，负责将原始的、带有噪声的轨迹点转化为符合物理规律的光滑曲线。
        核心思想是"在宏观匀速的假设下，敏锐捕捉真实的微观变速"。

        核心策略:
        1. 洛伦兹函数融合加权:
           - 建立一个"绝对匀速"的参考模型。
           - 计算二者偏差，应用洛伦兹函数 (1/(1+x^2)) 动态分配权重。
           - 偏差小 (Cruising) -> 权重趋近 1.0 -> 强力吸附在匀速线上 (极致平滑)。
           - 偏差大 (Braking)  -> 权重趋近 0.0 -> 快速逃逸并跟随观测值 (捕捉急刹)。

        2. 边缘透视补偿:
           - 问题: 车辆刚进入或离开画面边缘时，由于透视畸变，往往会出现系统性误差。
           - 方案: 对轨迹首尾两端 (约前/后25%) 应用正弦加权 (Sine-Ramp)，
             将边缘速度强制收敛至全过程的"平均速度"，消除进出场的跳变噪声。

        3. 宽窗微分加速度:
           - 问题: 直接对速度求导 (30 fps 时 dt=0.033s) 会极度放大高频噪声，导致加速度剧烈抖动。
           - 方案: 采用跨度为9帧的中心差分窗口。
           - 公式: a[i] = (v[i+k] - v[i-k]) / (t[i+k] - t[i-k])
           - 效果: 物理上等效于计算稍大时间窗口内的平均加速度，彻底抹平瞬时抖动，且保留了真实的加减速趋势。

        Args:
            trajectory (list): 原始轨迹点列表，包含 ['raw_x', 'raw_y', 'frame_id']。
            class_id (int): 车型ID，用于确定物理加速度上限 (如卡车 2.0m/s²)。

        Returns:
            list: 包含 ['speed', 'accel', 'op_mode'...] 的重构后轨迹列表。
        """
        # Step 1. 数据准备和预处理
        # --- 放弃处理过短的轨迹 ---
        if len(trajectory) < 5: return trajectory

        # --- 物理加速度限幅 ---
        ACCEL_LIMITS = {
            self.cfg.YOLO_CLASS_CAR: 5.0,
            self.cfg.YOLO_CLASS_BUS: 2.5,
            self.cfg.YOLO_CLASS_TRUCK: 2.0
        }
        phys_limit = ACCEL_LIMITS.get(class_id, 5.0)
        dt = 1.0 / self.cfg.FPS
        
        # --- 提取原始坐标 ---
        raw_x = np.array([p['raw_x'] for p in trajectory])
        raw_y = np.array([p['raw_y'] for p in trajectory])
        n_points = len(raw_x)

        # --- 内部辅助函数: 双向卷积平滑 (零相位滞后) ---
        def bidirectional_smooth(data, window):
            if len(data) < window: window = len(data) if len(data) % 2 == 1 else len(data) - 1
            if window < 3: return data
            pad_width = window // 2
            padded = np.pad(data, (pad_width, pad_width), mode='edge')
            kernel = np.ones(window) / window
            fwd = np.convolve(padded, kernel, mode='valid')
            padded_rev = np.pad(data[::-1], (pad_width, pad_width), mode='edge')
            bwd = np.convolve(padded_rev, kernel, mode='valid')[::-1]
            return (fwd + bwd) / 2.0

        # --- 基础位置平滑 ---
        pos_window = 31 
        smooth_x = bidirectional_smooth(raw_x, window=pos_window)
        smooth_y = bidirectional_smooth(raw_y, window=pos_window)
        
        # Step 2. 磁性融合核心逻辑
        # --- 绝对匀速模型 (Magnet) ---
        t_axis = np.arange(n_points) * dt
        coeff_x = np.polyfit(t_axis, smooth_x, 1) # 线性拟合
        coeff_y = np.polyfit(t_axis, smooth_y, 1)
        const_speed_val = np.sqrt(coeff_x[0]**2 + coeff_y[0]**2) 
        speed_linear = np.full(n_points, const_speed_val)
        
        # --- 局部观测速度 (Reality) ---
        grads_x = np.gradient(smooth_x, dt)
        grads_y = np.gradient(smooth_y, dt)
        inst_speed = np.sqrt(grads_x**2 + grads_y**2)
        speed_local = bidirectional_smooth(inst_speed, window=21) # 可缩短窗口以提升对变速事件的响应灵敏度
        
        # --- 计算洛伦兹权重 ---
        deviation = np.abs(speed_local - speed_linear)
        MAGNET_SIGMA = 1.0 # 逃逸阈值: 1.0 m/s (超过此值即视为真实变速)
        magnet_weight = 1.0 / (1.0 + (deviation / MAGNET_SIGMA) ** 2)
        
        # --- 加权融合 ---
        corrected_speed = magnet_weight * speed_linear + (1.0 - magnet_weight) * speed_local

        # Step 3. 边缘透视补偿 (Fade-in)
        # 解决车辆刚进入画面时因透视关系导致的速度虚高
        path_len = np.sum(np.sqrt(np.diff(smooth_x)**2 + np.diff(smooth_y)**2))
        duration = (n_points - 1) * dt
        avg_speed = path_len / duration if duration > 0 else 0
        
        if avg_speed > 1.5 and n_points > self.cfg.FPS * 1.5: 
            EDGE_RATIO = 0.25 
            weights = np.ones(n_points)
            fade_len = int(n_points * EDGE_RATIO)
            if fade_len > 0:
                ramp = np.linspace(0, 1, fade_len)
                fade_curve = 0.0 + (1.0 - 0.0) * np.sin(ramp * np.pi / 2)
                weights[:fade_len] = fade_curve
                weights[-fade_len:] = fade_curve[::-1]
            corrected_speed = weights * corrected_speed + (1 - weights) * avg_speed

        # Step 4. 加速度计算 (宽窗微分)
        # 使用稍窄的窗口以捕捉瞬时急刹 (k=9)
        k = 9  
        dense_accel = np.zeros(n_points)
        
        for i in range(n_points):
            idx_start = max(0, i - k)
            idx_end = min(n_points - 1, i + k)
            dv = corrected_speed[idx_end] - corrected_speed[idx_start]
            dt_span = (idx_end - idx_start) * dt
            
            if dt_span > 1e-4:
                val = dv / dt_span
                dense_accel[i] = np.clip(val, -phys_limit, phys_limit)
            else:
                dense_accel[i] = 0.0

        final_accel = bidirectional_smooth(dense_accel, window=31)

        # Step 5. 回写结果
        for i, p in enumerate(trajectory):
            p['rt_speed'] = float(inst_speed[i]) 
            p['rt_accel'] = float(dense_accel[i])
            p['speed'] = float(corrected_speed[i])
            p['accel'] = float(final_accel[i])

        return trajectory

    def _clean_opmode_sequence(self, opmodes):
        """
        [数据清洗] OpMode 序列状态机
        
        基于物理约束对工况序列进行后处理，消除不合理的跳变。
        
        规则:
        1. 消除短时毛刺 (Spike Removal)。
        2. 禁止非法转移 (e.g., Brake -> Hard Accel 直接跳变)。
        """
        if not opmodes: return []
        
        cleaned = np.array(opmodes, dtype=int)
        n = len(cleaned)
        
        # 策略 1: 消除单帧毛刺 (A-B-A -> A-A-A)
        for i in range(1, n - 1):
            prev, curr, next_ = cleaned[i-1], cleaned[i], cleaned[i+1]
            if prev == next_ and curr != prev:
                cleaned[i] = prev

        # 策略 2: 强制平滑过渡
        # 物理事实: 刹车(0) 不能瞬间变为 急加速(37)，中间必须经过缓加速
        for i in range(1, n):
            prev, curr = cleaned[i-1], cleaned[i]
            if prev == 0 and curr == 37:
                cleaned[i] = 35 
            
        return cleaned.tolist()

    def cleanup(self, final_frame_id):
        print("\n[Engine] 正在清理资源...")
        self.camera.stop()  # 停止 GStreamer 管道
        
        print("[Engine] 保存剩余车辆数据...")
        self._handle_exits(final_frame_id + 1000)
        self.db.close()
