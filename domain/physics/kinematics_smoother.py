import numpy as np
from scipy.signal import savgol_filter

class KinematicsSmoother:
    """
    [领域服务] 运动学平滑器
    升级为：基于绝对时间戳的动态时域降采样 (Time-based Decimation)。
    完美免疫边缘设备因算力瓶颈导致的变帧率(VFR)和随机掉帧问题。
    """
    def __init__(self, max_window: int = 15, polyorder: int = 2, target_fps: float = 5.0):
        self.max_window = max_window
        self.polyorder = polyorder
        self.target_fps = target_fps
        self.target_dt = 1.0 / target_fps  # 目标时间间隔 (如 5FPS 对应 0.2 秒)

    def process_1d(self, timestamps: np.ndarray, raw_x: np.ndarray, raw_y: np.ndarray):
        n_points = len(timestamps)
        if n_points < 3:
            return raw_x, raw_y, np.zeros(n_points), np.zeros(n_points)

        # 1. 核心横向降维
        mean_x = float(np.mean(raw_x))
        smoothed_x = np.full(n_points, mean_x)

        # 2. 基于绝对时间戳的动态抽样
        downsampled_indices = [0]  # 强制保留起点
        last_t = timestamps[0]

        for i in range(1, n_points):
            # 只有当距离上一个采样点的时间差达到目标间隔时，才进行采摘
            if timestamps[i] - last_t >= self.target_dt:
                downsampled_indices.append(i)
                last_t = timestamps[i]

        # 安全保留离场点，消除了可能出现的过小采样间隔
        if downsampled_indices[-1] != n_points - 1:
            # 计算最后一点与当前已采样列表中最后一个点的时间差
            dt_to_last = timestamps[n_points - 1] - timestamps[downsampled_indices[-1]]
            
            # 设定安全间距阈值
            safe_dt_threshold = self.target_dt * 0.5 
            
            if dt_to_last < safe_dt_threshold:
                # 如果采样点够多，直接用真实的离场点替换掉倒数第一个采样点
                if len(downsampled_indices) > 1:
                    downsampled_indices[-1] = n_points - 1
            else:
                # 间距足够大：直接追加离场点
                downsampled_indices.append(n_points - 1)

        # 判断抽样后的点数是否足够进行滤波
        if len(downsampled_indices) >= 5:
            ts_down = timestamps[downsampled_indices]
            y_down = raw_y[downsampled_indices]
        else:
            # 如果车辆极速驶过（总时间极短），则放弃降采样，保留全量高频数据
            ts_down = timestamps
            y_down = raw_y

        n_down = len(ts_down)
        window_length = min(self.max_window, n_down if n_down % 2 != 0 else n_down - 1)

        if window_length < 3:
            return smoothed_x, raw_y, np.zeros(n_points), np.zeros(n_points)

        # 3. 在时间均匀的降频轨迹上执行 S-G 滤波
        sm_y_down = savgol_filter(y_down, window_length=window_length, polyorder=self.polyorder)

        # 4. 计算一维速度
        speeds_down = np.zeros(n_down)
        for i in range(1, n_down):
            dy = sm_y_down[i] - sm_y_down[i-1]
            dt = ts_down[i] - ts_down[i-1]
            speeds_down[i] = abs(dy) / dt if dt > 1e-4 else speeds_down[i-1]
        speeds_down[0] = speeds_down[1] if n_down > 1 else 0.0

        # 速度二次平滑
        sm_speeds_down = savgol_filter(speeds_down, window_length=window_length, polyorder=self.polyorder)

        # 5. 计算加速度
        accels_down = np.zeros(n_down)
        for i in range(1, n_down - 1):
            dv = sm_speeds_down[i+1] - sm_speeds_down[i-1]
            dt = ts_down[i+1] - ts_down[i-1]
            accels_down[i] = dv / dt if dt > 1e-4 else 0.0
        
        if n_down > 1:
            dt_start = ts_down[1] - ts_down[0]
            accels_down[0] = (sm_speeds_down[1] - sm_speeds_down[0]) / dt_start if dt_start > 1e-4 else 0.0
            dt_end = ts_down[-1] - ts_down[-2]
            accels_down[-1] = (sm_speeds_down[-1] - sm_speeds_down[-2]) / dt_end if dt_end > 1e-4 else 0.0

        # 6. 插值还原：将低频骨架完美贴合回变帧率的原始时间轴
        # np.interp 天然支持非均匀采样的 X 轴，它会根据实际时间戳精准映射
        smoothed_y = np.interp(timestamps, ts_down, sm_y_down)
        smoothed_speeds = np.interp(timestamps, ts_down, sm_speeds_down)
        accels = np.interp(timestamps, ts_down, accels_down)

        return smoothed_x, smoothed_y, smoothed_speeds, accels
