import numpy as np
import math
from collections import defaultdict
import infra.config.loader as cfg

class VehicleRegistry:
    """
    [业务层] 车辆注册表
    负责维护所有在场车辆的生命周期、状态累积和轨迹记录。
    并不直接处理图像，而是处理由 MonitorEngine 传入的数据对象。
    """
    def __init__(self, target_fps: int = 30, min_survival_sec: float = 1.0, exit_timeout_sec: float = 3.0,
                 min_valid_pts: int = 15, min_moving_dist: float = 2.0):
        self.records = {}
        self.target_fps = target_fps
        
        # 生命周期配置
        self.min_survival_sec = min_survival_sec  # 最小存活秒数 (过滤误检)
        self.exit_timeout_sec = exit_timeout_sec  # 消失多少秒后认定为离场
        self.force_delay_sec = cfg.get("time_windows", {}).get("alignment_delay_sec", 60.0) # 延迟对齐时间，作为切片结算的基准
        
        # 数据质量配置
        self.min_valid_trajectory_points = min_valid_pts # 最小有效轨迹点数
        self.min_moving_distance_m = min_moving_dist     # 最小移动距离 (过滤静止车辆)

    def update(self, detections, frame_id, timestamp, model=None, roi_bounds=None):
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
            
            # --- 基础面积计算 ---
            width = max(0.0, float(box[2] - box[0]))
            height = max(0.0, float(box[3] - box[1]))
            area = width * height

            # 采用对数平滑 (Logarithmic Smoothing) 为像素面积加权
            # 原因：更符合神经网络感受野的信息边际递减规律
            # 加 1.0 是防止面积为 0 时 log 报错
            visual_quality = math.log(area + 1.0)

            # --- 空间与视角惩罚加权 (Exponential Spatial Weighting) ---
            spatial_multiplier = 1.0
            if roi_bounds and roi_bounds[1] > roi_bounds[0]:
                min_y, max_y = roi_bounds
                y_bottom = float(box[3])  # 取车辆检测框的底边 Y 坐标 (最能代表车辆在路面的真实位置)
                
                # 计算车辆在 ROI 垂直方向的相对进度 (0.0 ~ 1.0)
                # 0.0 表示在 ROI 最上方，1.0 表示到达 ROI 最下方的最佳抓拍底边
                progress = (y_bottom - min_y) / (max_y - min_y)
                progress = max(0.0, min(1.0, progress)) # 限制在 0 到 1 之间
                
                # 指数级放大系数 (alpha)
                # alpha=3.0 时，底边(最佳观测区)的单帧投票权重是顶边(畸变区)的 e^3 ≈ 20.08 倍
                alpha = 3.0 
                spatial_multiplier = math.exp(alpha * progress)
                
            # 最终复合权重
            # Conf (视觉模型置信度) * Visual Quality (视觉信息量) * Spatial Multiplier (观测位置增益)
            weight = conf * visual_quality * spatial_multiplier
            
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
                dt = 1.0 / self.target_fps # 回退到默认固定帧率
                
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
        检查哪些车辆已经离开画面（超时未更新），或者在画面中滞留过久触发强制临时结算。
        :return: list of (tid, record)
        """
        exits = []
        # 使用 list 包装 items() 以防在迭代中删除元素
        for tid, record in list(self.records.items()):
            # 1. 正常离场判定
            time_since_last_seen = current_timestamp - record['last_seen_time']
            is_timeout = time_since_last_seen > self.exit_timeout_sec
            
            # 2. 长时间赖场强制结算判定 (兜底策略)
            last_settled = record.get('last_settled_time', record['first_time'])
            time_since_settle = current_timestamp - last_settled
            is_forced = (not is_timeout) and (time_since_settle >= self.force_delay_sec)
            
            if is_timeout or is_forced:
                # 检查存活期、质量和移动距离
                life_span = record['last_seen_time'] - record['first_time']
                has_survival = life_span >= self.min_survival_sec
                
                min_valid_pts = getattr(self, 'min_valid_trajectory_points', 15)
                valid_samples = record.get('valid_samples_count', 0)
                has_quality = valid_samples >= min_valid_pts
                
                min_dist = getattr(self, 'min_moving_distance_m', 2.0)
                total_dist = record.get('total_distance_m', 0.0)
                has_movement = total_dist >= min_dist
                
                # 只要满足条件，无论是真离场还是被强制临时结算，都将其输出
                if has_survival and has_quality and has_movement:
                    # 构造浅拷贝对象，避免后续的重置操作污染准备落盘的数据
                    exit_record = dict(record)
                    exit_record['trajectory'] = list(record['trajectory'])
                    exit_record['op_mode_stats'] = dict(record['op_mode_stats'])
                    exit_record['exit_type'] = 'continued' if is_forced else 'exited'
                    exits.append((tid, exit_record))

                if is_timeout:
                    # 真正离场，彻底清除内存
                    del self.records[tid]
                elif is_forced:
                    # --- 强制临时结算，开始处理下一个切片 ---
                    # 1. 保留少量历史点，防止运动学 S-G 滤波器在切片衔接处产生跳变断层
                    keep_pts = 15  
                    self.records[tid]['trajectory'] = self.records[tid]['trajectory'][-keep_pts:]
                    self.records[tid]['valid_samples_count'] = len(self.records[tid]['trajectory'])
                    
                    # 2. 更新下一段切片的起始时间点（无缝衔接）
                    self.records[tid]['first_time'] = record['last_seen_time']
                    self.records[tid]['first_frame'] = record['last_seen_frame']
                    self.records[tid]['last_settled_time'] = current_timestamp
                    
                    # 3. 清空宏观累加器，防止下一切片的 Veh_Sum 产生重复统计
                    self.records[tid]['total_distance_m'] = 0.0
                    self.records[tid]['speed_sum'] = 0.0
                    self.records[tid]['speed_count'] = 0
                    self.records[tid]['max_speed'] = 0.0
                    self.records[tid]['brake_emission_mg'] = 0.0
                    self.records[tid]['tire_emission_mg'] = 0.0
                    self.records[tid]['op_mode_stats'].clear()
                    
                    # 不清除 class_votes(车型投票) 和 plate_history(车牌)，保持其身份属性稳定！
                    
        return exits

    def get_history(self, tid):
        return self.records.get(tid, {}).get('plate_history', [])
    
    def get_record(self, tid):
        return self.records.get(tid)
