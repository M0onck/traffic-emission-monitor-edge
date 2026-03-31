import numpy as np
from scipy.signal import savgol_filter

class KinematicsSmoother:
    """
    [领域服务] 运动学平滑器
    负责将带有噪声的原始物理轨迹，通过 1D 降维和 S-G 滤波推导出极高精度的物理量。
    可供在线 UI 引擎和离线解析器复用。
    """
    def __init__(self, max_window: int = 15, polyorder: int = 2):
        self.max_window = max_window
        self.polyorder = polyorder

    def process_1d(self, timestamps: np.ndarray, raw_x: np.ndarray, raw_y: np.ndarray):
        """
        执行 1D 降维平滑及运动学重构
        返回: smoothed_x, smoothed_y, speeds, accels
        """
        n_points = len(timestamps)
        if n_points < 3:
            return raw_x, raw_y, np.zeros(n_points), np.zeros(n_points)

        # 1. 核心降维：计算横向位置均值，强制对齐到虚拟轨道
        mean_x = float(np.mean(raw_x))
        window_length = min(self.max_window, n_points if n_points % 2 != 0 else n_points - 1)

        if window_length < 3:
            smoothed_x = np.full(n_points, mean_x)
            return smoothed_x, raw_y, np.zeros(n_points), np.zeros(n_points)

        # 2. 单维平滑：仅对纵向 Y 轴进行 S-G 平滑
        smoothed_y = savgol_filter(raw_y, window_length=window_length, polyorder=self.polyorder)

        # 3. 极简一维速度计算
        speeds = np.zeros(n_points)
        for i in range(1, n_points):
            dy = smoothed_y[i] - smoothed_y[i-1]
            dt = timestamps[i] - timestamps[i-1]
            speeds[i] = abs(dy) / dt if dt > 1e-4 else speeds[i-1]
        speeds[0] = speeds[1] if n_points > 1 else 0.0

        # 4. 速度二次平滑
        smoothed_speeds = savgol_filter(speeds, window_length=window_length, polyorder=self.polyorder)

        # 5. 中心差分计算加速度
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

        # 生成 1D 降维后的恒定 X 数组
        smoothed_x = np.full(n_points, mean_x)
        
        return smoothed_x, smoothed_y, smoothed_speeds, accels
