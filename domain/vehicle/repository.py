import numpy as np
import math
from collections import defaultdict

class VehicleRegistry:
    """
    [业务层] 车辆注册表
    负责维护所有在场车辆的生命周期、状态累积和轨迹记录。
    并不直接处理图像，而是处理由 MonitorEngine 传入的数据对象。
    """
    def __init__(self, target_fps: int = 30, min_survival_sec: float = 1.0, exit_timeout_sec: float = 1.0,
                 min_valid_pts: int = 15, min_moving_dist: float = 2.0):
        self.records = {}
        self.target_fps = target_fps
        
        # 生命周期配置
        self.min_survival_frames = min_survival_sec  # 最小存活秒数 (过滤误检)
        self.exit_threshold = exit_timeout_sec       # 消失多少秒后认定为离场
        
        # 数据质量配置
        self.min_valid_trajectory_points = min_valid_pts # 最小有效轨迹点数
        self.min_moving_distance_m = min_moving_dist     # 最小移动距离 (过滤静止车辆)

    def update(self, detections, frame_id, timestamp, model=None):
        """
        根据 YOLO 检测结果更新车辆列表
        更新逻辑：
        1. 注册新 ID
        2. 更新活跃状态 (last_seen)
        3. 投票确认车型 (Class Voting)
        """
        # 双重保险：拦截空检测，防止无效计算
        if detections is None or detections.tracker_id is None or len(detections.tracker_id) == 0:
            return

        for tid, cid, conf, box in zip(
                detections.tracker_id, 
                detections.class_id, 
                detections.confidence,
                detections.xyxy
            ):
            cid = int(cid)
            conf = float(conf)
            
            # 使用面积加权置信度进行投票
            width = max(0.0, float(box[2] - box[0]))
            height = max(0.0, float(box[3] - box[1]))
            area = width * height
            weight = conf * math.sqrt(area) 
            
            # 动态获取类名，防止 model 为 None 时引发 AttributeError
            class_name = model.names[cid] if model and hasattr(model, 'names') else f"Class_{cid}"

            if tid not in self.records:
                self.records[tid] = {
                    'class_id': cid,
                    'class_name': class_name,
                    'class_votes': defaultdict(float),
                    'trajectory': [],
                    'valid_samples_count': 0,
                    'first_frame': frame_id,
                    'first_time': timestamp,      # 记录首次出现的绝对时间
                    'max_conf': float(conf),
                    'last_seen_frame': frame_id,
                    'last_seen_time': timestamp,  # 记录最后一次出现的绝对时间
                    'reported': False,
                    'plate_history': [],
                    # 宏观统计累积槽
                    'op_mode_stats': defaultdict(int),
                    'brake_emission_mg': 0.0,
                    'tire_emission_mg': 0.0,
                    'max_speed': 0.0,
                    'speed_sum': 0.0,
                    'speed_count': 0,
                    'total_distance_m': 0.0
                }
            
            rec = self.records[tid]
            rec['last_seen_frame'] = frame_id
            rec['last_seen_time'] = timestamp     # 每次更新检测框时刷新绝对时间
            rec['class_votes'][cid] += weight
            
            # 更新主要车型判定 (Majority Vote)
            best_class = max(rec['class_votes'], key=rec['class_votes'].get)
            rec['class_id'] = best_class
            
            # 更新投票处，同样进行安全获取
            rec['class_name'] = model.names[best_class] if model and hasattr(model, 'names') else f"Class_{best_class}"

            if conf > rec['max_conf']:
                rec['max_conf'] = conf

    def append_kinematics(self, tid, frame_id, speed, accel, raw_x=None, raw_y=None, pixel_x=None, pixel_y=None, timestamp=None):
        """
        记录单帧运动学数据
        :param tid: Tracker ID
        :param frame_id: 当前帧号
        :param speed: 实时速度 (m/s)
        :param accel: 实时加速度 (m/s^2)
        :param raw_x: 物理坐标 X (经过逆透视变换/平滑后的值)
        :param raw_y: 物理坐标 Y (经过逆透视变换/平滑后的值)
        :param pixel_x: 原始像素坐标 X (用于几何测距)
        :param pixel_y: 原始像素坐标 Y (用于几何测距)
        :param timestamp: 当前帧的精确绝对时间戳 (NTP对齐)
        """
        if tid in self.records:
            rec = self.records[tid]
            
            # 存入轨迹列表
            rec['trajectory'].append({
                'frame_id': frame_id,
                'speed': speed,
                'accel': accel,
                'raw_x': raw_x,
                'raw_y': raw_y,
                'pixel_x': pixel_x,
                'pixel_y': pixel_y,
                'timestamp': timestamp  # 保存精确时间戳
            })
            rec['valid_samples_count'] = rec.get('valid_samples_count', 0) + 1

            # --- 动态 dt 距离估算 (用于静止车辆过滤) ---
            # 如果传入了时间戳，并且轨迹中至少有两个点，则计算真实的动态 dt
            if timestamp is not None and len(rec['trajectory']) >= 2:
                dt = timestamp - rec['trajectory'][-2]['timestamp']
                # 避免由于数据异常导致 dt 出现负数或过大
                dt = max(0.001, min(dt, 0.5)) 
            else:
                dt = 1.0 / self.fps # 回退到默认固定帧率
                
            rec['total_distance_m'] = rec.get('total_distance_m', 0.0) + (speed * dt)
            
            # 统计最大速度和平均速度辅助数据
            if speed > rec['max_speed']:
                rec['max_speed'] = speed
            rec['speed_sum'] += speed
            rec['speed_count'] += 1

    def accumulate_opmode(self, record, op_mode: int):
        """累积 OpMode 统计 (用于宏观表)"""
        record['op_mode_stats'][op_mode] += 1

    def accumulate_brake_emission(self, record, mass_mg: float):
        """累积刹车排放 (用于宏观表)"""
        record['brake_emission_mg'] += mass_mg

    def accumulate_tire_emission(self, record, mass_mg: float):
        """累积轮胎排放 (用于宏观表)"""
        record['tire_emission_mg'] += mass_mg

    # 兼容旧接口的别名
    def update_emission_stats(self, record, op_mode, brake_emission):
        self.accumulate_opmode(record, op_mode)
        self.accumulate_brake_emission(record, brake_emission)

    def update_tire_stats(self, record, tire_emission):
        self.accumulate_tire_emission(record, tire_emission)

    def add_plate_history(self, tid, color, area, conf):
        """记录一次 OCR 识别结果，用于后续投票"""
        if tid in self.records:
            self.records[tid]['plate_history'].append({
                'color': color, 'area': area, 'conf': conf
            })

    def check_exits(self, frame_id, current_timestamp):
        """
        检查哪些车辆已经离开画面（超时未更新）。
        :return: list of (tid, record)
        """
        timed_out_ids = []
        for tid, record in self.records.items():
            # 使用时间戳差值判定超时 (例如：超过 1.0 秒未见)
            if current_timestamp - record['last_seen_time'] > self.exit_timeout_sec:
                timed_out_ids.append(tid)

        valid_exits = []
        for tid in timed_out_ids:
            record = self.records[tid]
            
            # 1. 存活时间过滤
            life_span = record['last_seen_time'] - record['first_time']
            has_survival = life_span >= self.min_survival_sec
            
            # 2. 有效点数过滤
            min_valid_pts = getattr(self, 'min_valid_trajectory_points', 15)
            valid_samples = record.get('valid_samples_count', 0)
            has_quality = valid_samples >= min_valid_pts
            
            # 3. 移动距离过滤 (防止静止误检)
            min_dist = getattr(self, 'min_moving_distance_m', 2.0)
            total_dist = record.get('total_distance_m', 0.0)
            has_movement = total_dist >= min_dist
            
            if has_survival and has_quality and has_movement:
                valid_exits.append((tid, record))

            # 无论是否有效，都从内存中移除
            del self.records[tid]
            
        return valid_exits

    def get_history(self, tid):
        return self.records.get(tid, {}).get('plate_history', [])
    
    def get_record(self, tid):
        return self.records.get(tid)
