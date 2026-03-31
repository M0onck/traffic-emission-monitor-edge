import cv2
import numpy as np
import supervision as sv
import time
from collections import defaultdict
from ui.renderer import resize_with_pad, LabelData
from infra.time.ntp_sync import TimeSynchronizer
from scipy.signal import savgol_filter

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

        # 分类器引用更新：原先完整的 OCR 已被替换为仅针对车牌属性的小模型分类器
        self.plate_classifier = components.get('plate_classifier') 
        
        # 修复潜在的 AttributeError: 恢复对基础车型分类器(用于解析车辆类型逻辑)的引用
        self.classifier = components.get('classifier')

        # 初始化 Python 层的高性能 ByteTrack 追踪器
        # lost_track_buffer 设为 30 帧，足以抗遮挡；配合 Python 先进的卡尔曼预测，能完美解决高速幽灵框
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25, 
            lost_track_buffer=30, 
            minimum_matching_threshold=0.8,
            frame_rate=config.FPS
        )

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
                        self.comps['transformer'] = ViewTransformer(pts, self.comps['target_points'])

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
        
        # --- Step 1: 解析 Hailo Metadata (从底层接收推理结果) ---
        xyxy, class_ids, confs = [], [], []

        try:
            roi = hailo.get_roi_from_buffer(buffer)
            hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
            
            for det in hailo_detections:
                # 获取检测框
                bbox = det.get_bbox()
                raw_x1, raw_y1 = int(bbox.xmin() * w), int(bbox.ymin() * h)
                raw_x2, raw_y2 = int(bbox.xmax() * w), int(bbox.ymax() * h)
                
                # 安全过滤：确保 x1 < x2 且 y1 < y2，防止硬件输出异常坐标导致下游崩溃
                x1, x2 = min(raw_x1, raw_x2), max(raw_x1, raw_x2)
                y1, y2 = min(raw_y1, raw_y2), max(raw_y1, raw_y2)
                
                # 获取标签，并进行严格的白名单过滤
                label = det.get_label()
                if label not in self.label_map:
                    continue  # 直接丢弃所有非 car/bus/truck 的目标（如 person, bicycle 等）
                    
                cid = self.label_map[label]
                
                xyxy.append([x1, y1, x2, y2])
                class_ids.append(cid)
                confs.append(det.get_confidence())

        except Exception as e:
            # 如果解析出错（或者在非真实设备环境运行），构建空的数据防止崩溃
            pass
            
        # 构建初步的 Detections (不再从 Hailo 接收 tracker_id)
        if len(xyxy) > 0:
            detections = sv.Detections(
                xyxy=np.array(xyxy, dtype=np.float32),
                confidence=np.array(confs, dtype=np.float32),
                class_id=np.array(class_ids, dtype=int)
            )
            
            # 先用 NMS 强力合并完美重叠的检测框 (去除多余的特征)
            detections = detections.with_nms(threshold=0.4, class_agnostic=True)
            
            # 把干净的、没有任何重叠的框喂给追踪器，再分配 ID
            detections = self.tracker.update_with_detections(detections)
            
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
        kinematics_data = {}
        
        if self.motion_on and self.comps.get('kinematics'):
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed = self.comps['transformer'].transform_points(points)
            roi_bounds = self.comps['transformer'].get_roi_vertical_bounds()
            
            kinematics_data = self.comps['kinematics'].update(
                detections, transformed, frame.shape, frame_timestamp, roi_y_range=roi_bounds
            )

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
                        pixel_x=raw_point[0], pixel_y=raw_point[1],
                        timestamp=frame_timestamp
                    )

        # --- Step 6: 可视化渲染 ---
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
        for tid, color_type, conf, rel_landmarks in results:
            # 恢复正常的配置文件阈值判断
            if conf > self.cfg.OCR_CONF_THRESHOLD: 
                self.registry.add_plate_history(tid, color_type, 1.0, conf)
                self.plate_cache[tid] = {
                    'color': color_type,
                    'rel_landmarks': rel_landmarks
                }

    def _handle_exits(self, frame_id):
        """
        处理离场车辆：执行最终结算、生成报表并入库。
        """
        for tid, record in self.registry.check_exits(frame_id):
            # Step 1. 解析最终属性 (车牌、细分车型)
            final_plate, final_type_str = self.classifier.resolve_type(
                record['class_id'], record.get('plate_history', [])
            )

            # 车牌纯置信度加权投票机制
            voted_color = "Unknown"
            history = record.get('plate_history', [])
            if history:
                scores = defaultdict(float)
                for entry in history:
                    # 直接累加置信度，置信度越高的帧权重越大
                    scores[entry['color']] += entry.get('conf', 1.0)
                # 选出累计置信度得分最高的颜色
                voted_color = max(scores, key=scores.get)
            
            # 将投票决定的最终颜色存入 record，确保下游数据库和 UI 拿到的是同一份数据
            record['final_plate_color'] = voted_color
            
            # Step 2. 几何距离重算
            self._recalculate_distance_geometric(record)

            # Step 3. 微观时空轨迹与 VSP 结算 (核心逻辑)
            if 'trajectory' in record:
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
        重构版：植入 Savitzky-Golay 非因果滤波，实现无相位滞后的全局轨迹与运动学平滑。
        """
        trajectory = record.get('trajectory', [])

        # Step 1. 轨迹头尾清洗 (Trimming)
        TRIM_SIZE = 5 
        if len(trajectory) > (TRIM_SIZE * 2 + 5):
            trajectory = trajectory[TRIM_SIZE : -TRIM_SIZE]
        else:
            return # 轨迹太短，放弃计算

        # Step 2. 非因果 Savitzky-Golay 全局平滑
        n_points = len(trajectory)
        if n_points >= 7:  # S-G 窗口大小设为 7
            # 提取原始绝对物理坐标与动态时间戳
            raw_x = np.array([p.get('raw_x', 0.0) for p in trajectory])
            raw_y = np.array([p.get('raw_y', 0.0) for p in trajectory])
            timestamps = np.array([p.get('timestamp', 0.0) for p in trajectory])

            # 动态调整窗口大小 (S-G filter 要求 window_length 必须是正奇数，且 <= n_points)
            window_length = min(7, n_points if n_points % 2 != 0 else n_points - 1)
            
            if window_length >= 3:
                # 2.1 对物理坐标进行一次 S-G 滤波去噪 (保峰并去除高频抖动)
                smoothed_x = savgol_filter(raw_x, window_length=window_length, polyorder=2)
                smoothed_y = savgol_filter(raw_y, window_length=window_length, polyorder=2)

                # 2.2 结合平滑后的坐标与动态 dt，严谨推导瞬时速度
                speeds = np.zeros(n_points)
                for i in range(1, n_points):
                    dx = smoothed_x[i] - smoothed_x[i-1]
                    dy = smoothed_y[i] - smoothed_y[i-1]
                    dt = timestamps[i] - timestamps[i-1]
                    speeds[i] = np.hypot(dx, dy) / dt if dt > 1e-4 else speeds[i-1]
                speeds[0] = speeds[1] if n_points > 1 else 0.0

                # 2.3 对速度序列进行二次 S-G 滤波，获取平滑且保物理极值的高质量曲线
                smoothed_speeds = savgol_filter(speeds, window_length=window_length, polyorder=2)

                # 2.4 中心差分计算无滞后的真实加速度
                accels = np.zeros(n_points)
                for i in range(1, n_points - 1):
                    dv = smoothed_speeds[i+1] - smoothed_speeds[i-1]
                    dt = timestamps[i+1] - timestamps[i-1]
                    accels[i] = dv / dt if dt > 1e-4 else 0.0
                
                if n_points > 1:
                    dt_start = timestamps[1] - timestamps[0]
                    accels[0] = (smoothed_speeds[1] - smoothed_speeds[0]) / dt_start if dt_start > 1e-4 else 0.0
                    dt_end = timestamps[-1] - timestamps[-2]
                    accels[-1] = (smoothed_speeds[-1] - smoothed_speeds[-2]) / dt_end if dt_end > 1e-4 else 0.0

                # 2.5 覆盖原始极度抖动的坐标和速度
                for i in range(n_points):
                    trajectory[i]['raw_x'] = float(smoothed_x[i])
                    trajectory[i]['raw_y'] = float(smoothed_y[i])
                    trajectory[i]['speed'] = float(smoothed_speeds[i])
                    trajectory[i]['accel'] = float(accels[i])
        
        record['trajectory'] = trajectory # 回写清洗并平滑后的轨迹

        # 暴露结算完毕的数据给 UI Dashboard 供抽样展示
        self.latest_exit_record = {
            'tid': tid,
            'record': record,
            'type_str': final_type_str
        }

        # Step 3. 物理特征提取与入库
        vsp_calc = self.comps.get('vsp_calculator')
        if not vsp_calc:
            return

        final_class_id = record['class_id']
        for point in trajectory:
            v = point['speed']
            a = point['accel']
            fid = point['frame_id']
            
            vsp = vsp_calc.calculate(v, a, final_class_id)
            
            db_payload = {
                'timestamp': point.get('timestamp', 0.0),
                'ipm_x': point.get('raw_x', 0.0),
                'ipm_y': point.get('raw_y', 0.0),
                'speed': v, 
                'accel': a, 
                'vsp': vsp
            }
            self.db.insert_micro(fid, tid, db_payload)
            
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

        # 停止 NTP 时钟同步器的后台线程
        if hasattr(self, 'time_sync'):
            self.time_sync.stop()
            print("[Engine] 时钟同步守护线程已停止。")
        
        print("[Engine] 保存剩余车辆数据...")
        self._handle_exits(final_frame_id + 1000)
        self.db.close()
