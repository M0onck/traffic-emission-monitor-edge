import numpy as np
from scipy.signal import savgol_filter, medfilt

class KinematicsSmoother:
    """
    [领域服务] 运动学平滑器
    核心职责 1: 基于绝对时间戳与虚拟均匀时间轴的抗抖动物理平滑 (Anti-aliasing Smoothing)。
    核心职责 2: 提供安全的末端数据降采样工具，用于减轻数据库 I/O 负担 (Database Decimation)。
    """
    def __init__(self, target_fps: float = 30.0):
        self.target_fps = target_fps
        self.target_dt = 1.0 / target_fps
        self.polyorder = 2

    @staticmethod
    def get_downsampled_indices(timestamps: np.ndarray, target_dt: float) -> list:
        """
        [数据库入库专用] 降采样工具。
        注意：仅限用于经过充分低通滤波后的数据抽样，切勿用于滤波前的原始噪声数据。
        包含对离场点（尾点）安全间距的严谨边界处理，确保车辆离场时间精确落盘。
        """
        n_points = len(timestamps)
        if n_points == 0:
            return []
        if n_points < 3:
            return list(range(n_points))

        downsampled_indices = [0]  # 强制保留起点
        last_t = timestamps[0]

        for i in range(1, n_points):
            if timestamps[i] - last_t >= target_dt:
                downsampled_indices.append(i)
                last_t = timestamps[i]

        # 安全保留离场点
        if downsampled_indices[-1] != n_points - 1:
            dt_to_last = timestamps[n_points - 1] - timestamps[downsampled_indices[-1]]
            safe_dt_threshold = target_dt * 0.5 
            
            if dt_to_last < safe_dt_threshold:
                if len(downsampled_indices) > 1:
                    downsampled_indices[-1] = n_points - 1
            else:
                downsampled_indices.append(n_points - 1)

        return downsampled_indices

    def process_1d(self, timestamps: np.ndarray, raw_x: np.ndarray, raw_y: np.ndarray):
        """
        处理一维（主要为纵向 Y 轴）的高精度运动学平滑与求导。
        流水线：中值去刺 -> 绝对均匀时间重采样 -> 动态窗口 S-G 滤波 -> 原始时间轴还原。
        """
        n_points = len(timestamps)
        if n_points < 3:
            return raw_x, raw_y, np.zeros(n_points), np.zeros(n_points)

        mean_x = float(np.mean(raw_x))
        smoothed_x = np.full(n_points, mean_x)

        # ==========================================
        # 1. 中值滤波 (Median Filter)：专杀 YOLO 框跳动尖刺
        # ==========================================
        med_kernel = min(5, n_points if n_points % 2 != 0 else n_points - 1)
        if med_kernel >= 3:
            despiked_y = medfilt(raw_y, kernel_size=med_kernel) 
        else:
            despiked_y = raw_y

        # ==========================================
        # 2. 重采样 (Resampling)：建立绝对均匀虚拟时间轴
        # ==========================================
        t_start, t_end = timestamps[0], timestamps[-1]
        
        # 动态自适应 dt：防止画面切片时长极短时崩溃兜底
        target_dt = self.target_dt
        if (t_end - t_start) < target_dt * 5:
            target_dt = max(0.033, (t_end - t_start) / 5.0) 

        uniform_t = np.arange(t_start, t_end + target_dt, target_dt)
        n_uniform = len(uniform_t)

        uniform_y = np.interp(uniform_t, timestamps, despiked_y)

        # ==========================================
        # 3. S-G 物理拟合与求导 (保持约 1.5 秒恒定物理感受野)
        # ==========================================
        physical_window_sec = 1.5 
        desired_window_length = int(physical_window_sec / target_dt)
        
        window_length = min(desired_window_length, n_uniform if n_uniform % 2 != 0 else n_uniform - 1)
        if window_length % 2 == 0:
            window_length -= 1

        if window_length < 3:
            return smoothed_x, raw_y, np.zeros(n_points), np.zeros(n_points)

        # S-G 轨迹滤波
        sm_y_uniform = savgol_filter(uniform_y, window_length=window_length, polyorder=self.polyorder)

        # 速度求导 (m/s)
        vy_uniform = np.gradient(sm_y_uniform, target_dt)
        speeds_uniform = np.abs(vy_uniform)
        
        # 速度 S-G 二次平滑与加速度求导 (m/s^2)
        sm_speeds_uniform = savgol_filter(speeds_uniform, window_length=window_length, polyorder=self.polyorder)
        accels_uniform = np.gradient(sm_speeds_uniform, target_dt)

        # ==========================================
        # 4. 精准还原：映射回变帧率原始时间轴供 UI 与 DB 使用
        # ==========================================
        smoothed_y = np.interp(timestamps, uniform_t, sm_y_uniform)
        smoothed_speeds = np.interp(timestamps, uniform_t, sm_speeds_uniform)
        accels = np.interp(timestamps, uniform_t, accels_uniform)

        return smoothed_x, smoothed_y, smoothed_speeds, accels
