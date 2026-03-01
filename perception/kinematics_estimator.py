import numpy as np
from collections import deque

class LightweightKinematicsState:
    """
    [数学组件] 轻量级运动学状态追踪器
    取代原有的 KalmanFilterCV + LinearAccelEstimator，使用双层队列完成无延迟的差分计算。
    """
    def __init__(self, dt, speed_window=31):
        self.dt = dt
        # 用短窗口 (5帧) 计算速度，保证物理位移差分的灵敏度
        self.pos_history = deque(maxlen=5)
        # 用长窗口 (speed_window) 计算加速度，过滤高频导数噪声
        self.speed_history = deque(maxlen=speed_window)

    def update(self, current_pos_m):
        self.pos_history.append(current_pos_m)
        
        speed = 0.0
        accel = 0.0

        # 1. 计算当前速度 (物理坐标差分)
        if len(self.pos_history) >= 2:
            p_now = self.pos_history[-1]
            p_old = self.pos_history[0]
            dt_span = (len(self.pos_history) - 1) * self.dt
            
            # 计算欧氏距离位移
            dist = np.hypot(p_now[0] - p_old[0], p_now[1] - p_old[1])
            if dt_span > 1e-4:
                speed = dist / dt_span

        self.speed_history.append(speed)

        # 2. 计算当前加速度 (长窗口平滑差分)
        min_accel_window = 5
        if len(self.speed_history) >= min_accel_window:
            v_now = self.speed_history[-1]
            v_old = self.speed_history[0]
            dt_span_acc = (len(self.speed_history) - 1) * self.dt
            if dt_span_acc > 1e-4:
                accel = (v_now - v_old) / dt_span_acc

        return speed, accel, current_pos_m[0], current_pos_m[1]


class KinematicsEstimator:
    """
    [感知层] 运动学估算器 (边缘端极简版)
    移除卡尔曼滤波依赖，利用底层硬件生成的稳定边框，大幅释放 CPU 算力。
    """
    def __init__(self, config: dict):
        self.fps = config.get("fps", 30)
        self.dt = 1.0 / self.fps
        
        params = config.get("kinematics", {})
        self.speed_window = params.get("speed_window", 31) 
        self.border_margin = params.get("border_margin", 20)
        self.min_tracking_frames = params.get("min_tracking_frames", 10)
        self.max_physical_accel = params.get("max_physical_accel", 6.0)
        
        self.trackers = {}     
        self.active_frames = {}
        self.last_raw_pixels = {} 

    def update(self, detections, points_transformed, frame_shape, roi_y_range=None):
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
                self.trackers[tid] = LightweightKinematicsState(
                    dt=self.dt, 
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

            # 3. 核心计算：极简坐标差分
            tracker = self.trackers[tid]
            speed, accel, curr_x, curr_y = tracker.update(point)

            # 4. 静止归零过滤
            if not is_moving and speed < 1.0:
                speed = 0.0
                accel = 0.0
                tracker.speed_history.clear()

            # 5. 画面边界与有效追踪时长过滤
            box = raw_boxes[i]
            x1, y1, x2, y2 = box
            if (x1 < self.border_margin or y1 < self.border_margin or 
                x2 > img_w - self.border_margin or y2 > img_h - self.border_margin):
                continue 

            if self.active_frames[tid] < self.min_tracking_frames:
                continue 

            accel = np.clip(accel, -self.max_physical_accel, self.max_physical_accel)

            results[tid] = {
                "speed": speed, 
                "accel": accel,
                "curr_x": curr_x,
                "curr_y": curr_y
            }
            
        return results
