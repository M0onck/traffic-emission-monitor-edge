import numpy as np
from collections import deque

class LightweightKinematicsState:
    """
    [数学组件] 轻量级运动学状态追踪器 (变帧率/精确时间戳适配版)
    取代原有的固定帧率差分，使用双层队列并结合精确的真实时间戳 (Timestamp) 
    完成无延迟、自适应变帧率的差分计算。这对于处理网络抖动或硬件算力波动导致的视频掉帧尤为关键。
    """
    def __init__(self, speed_window=31):
        # 用短窗口 (5帧) 计算速度，保证物理位移差分的灵敏度
        # 队列中存储元组: (timestamp, x, y)
        self.pos_history = deque(maxlen=5)
        # 用长窗口 (speed_window) 计算加速度，过滤高频导数噪声
        # 队列中存储元组: (timestamp, speed)
        self.speed_history = deque(maxlen=speed_window)

    def update(self, current_pos_m, timestamp):
        """
        更新运动学状态并计算瞬时速度和加速度
        
        Args:
            current_pos_m (tuple): 当前帧车辆在世界坐标系下经逆透视变换(IPM)后的 (x, y) 坐标，单位: 米
            timestamp (float): 当前帧的精确时间戳 (绝对时间或相对时间)，单位: 秒
            
        Returns:
            tuple: (speed_m_s, accel_m_s2, curr_x, curr_y)
        """
        self.pos_history.append((timestamp, current_pos_m[0], current_pos_m[1]))
        
        speed = 0.0
        accel = 0.0

        # 1. 计算当前速度 (基于真实物理时间差分)
        if len(self.pos_history) >= 2:
            t_now, x_now, y_now = self.pos_history[-1]
            t_old, x_old, y_old = self.pos_history[0]
            
            # 计算动态 dt (距离队列头部最老记录的时间跨度)
            dt_span = t_now - t_old
            
            # 避免除以 0 导致崩溃 (例如：处理同一张图片、视频卡顿等异常情况)
            if dt_span > 1e-4:
                # 计算欧氏距离位移
                dist = np.hypot(x_now - x_old, y_now - y_old)
                speed = dist / dt_span

        self.speed_history.append((timestamp, speed))

        # 2. 计算当前加速度 (基于长窗口真实时间平滑差分)
        min_accel_window = 5
        if len(self.speed_history) >= min_accel_window:
            t_now, v_now = self.speed_history[-1]
            t_old, v_old = self.speed_history[0]
            
            # 同样获取头尾的速度时间跨度
            dt_span_acc = t_now - t_old
            
            if dt_span_acc > 1e-4:
                accel = (v_now - v_old) / dt_span_acc

        return speed, accel, current_pos_m[0], current_pos_m[1]


class KinematicsEstimator:
    """
    [感知层] 运动学估算器 (边缘端极简版)
    移除对固定 FPS 的依赖和卡尔曼滤波，利用底层硬件生成的稳定边框和精准时间戳，
    大幅释放树莓派 CPU 算力的同时保证物理测算的准确性，为后续源解析透传精准原始时序数据。
    """
    def __init__(self, config: dict):
        # 移除 fps 和 dt，这不再是固定配置
        params = config.get("kinematics", {})
        self.speed_window = params.get("speed_window", 31) 
        self.border_margin = params.get("border_margin", 20)
        self.min_tracking_frames = params.get("min_tracking_frames", 10)
        self.max_physical_accel = params.get("max_physical_accel", 6.0)
        
        self.trackers = {}     
        self.active_frames = {}
        self.last_raw_pixels = {} 

    def update(self, detections, points_transformed, frame_shape, frame_timestamp, roi_y_range=None):
        """
        更新当前帧所有检测目标的运动学状态
        
        Args:
            detections: YOLO/追踪器输出的检测结果 (包含 raw_boxes 和 tracker_id)
            points_transformed: 经过 IPM 逆透视变换后的物理坐标点阵
            frame_shape: 图像尺寸 (height, width, channels)
            frame_timestamp (float): 当前视频帧的高精度时间戳 (需要从管道/NTP透传进来)
            roi_y_range: 感兴趣区域 (可选)
        """
        results = {}
        img_h, img_w = frame_shape[:2]
        
        raw_boxes = detections.xyxy
        if len(raw_boxes) > 0:
            raw_centers_x = (raw_boxes[:, 0] + raw_boxes[:, 2]) / 2
            raw_centers_y = (raw_boxes[:, 1] + raw_boxes[:, 3]) / 2
        else:
            raw_centers_x, raw_centers_y = [], []
        
        for i, (tid, point) in enumerate(zip(detections.tracker_id, points_transformed)):
            # 1. 状态初始化
            if tid not in self.trackers:
                # 动态 dt 模式下，不需要传入 dt
                self.trackers[tid] = LightweightKinematicsState(
                    speed_window=self.speed_window
                )
                self.active_frames[tid] = 0
                self.last_raw_pixels[tid] = (raw_centers_x[i], raw_centers_y[i])
            
            self.active_frames[tid] += 1
            
            # 2. 像素级静止检测 (抑制轻微形变引起的抖动)
            curr_pixel = (raw_centers_x[i], raw_centers_y[i])
            prev_pixel = self.last_raw_pixels.get(tid, curr_pixel)
            pixel_dist = np.hypot(curr_pixel[0]-prev_pixel[0], curr_pixel[1]-prev_pixel[1])
            self.last_raw_pixels[tid] = curr_pixel
            is_moving = pixel_dist > 0.5 

            # 3. 核心计算：传入精确时间戳进行变帧率坐标/速度求导
            tracker = self.trackers[tid]
            speed, accel, curr_x, curr_y = tracker.update(point, frame_timestamp)

            # 4. 静止归零过滤 (去除漂移速度)
            if not is_moving and speed < 1.0:
                speed = 0.0
                accel = 0.0
                tracker.speed_history.clear()

            # 5. 画面边界与有效追踪时长过滤
            # 处于画面边缘或刚出现的车辆容易引起透视畸变导致测速异常，剔除该部分数据
            box = raw_boxes[i]
            x1, y1, x2, y2 = box
            if (x1 < self.border_margin or y1 < self.border_margin or 
                x2 > img_w - self.border_margin or y2 > img_h - self.border_margin):
                continue 

            if self.active_frames[tid] < self.min_tracking_frames:
                continue 

            # 6. 加速度物理极限裁剪 (消除异常跳变，例如检测框突然变化)
            accel = np.clip(accel, -self.max_physical_accel, self.max_physical_accel)

            results[tid] = {
                "speed": speed, 
                "accel": accel,
                "curr_x": curr_x,
                "curr_y": curr_y
            }
            
        return results
