import math
import numpy as np
from collections import defaultdict
import plotext as plt

class Reporter:
    def __init__(self, config: dict):
        """
        初始化报告器
        """
        self.debug_mode = config.get('debug_mode', False)
        self.fps = config.get('fps', 30)
        self.min_survival_frames = config.get('min_survival_frames', 30)

    def print_exit_report(self, tid, record, kinematics_estimator, vehicle_classifier):
        """
        [输出] 车辆离场报告
        重构版：仅包含基础信息、OCR结果、物理统计(速度/里程)与运动学曲线图。
        """
        if not self.debug_mode: return

        # 1. 存活时间过滤
        life_span = record.get('last_seen_frame', 0) - record.get('first_frame', 0)
        if life_span < self.min_survival_frames: return 

        duration = life_span / self.fps

        # 2. 车牌/车型投票解析
        history = record.get('plate_history', [])
        final_plate = "Unknown"
        vote_info = "No OCR"
        
        if history:
            scores = defaultdict(float)
            total_weight = 0.0
            for entry in history:
                w = entry.get('conf', 1.0) * math.sqrt(entry.get('area', 0.0))
                scores[entry['color']] += w
                total_weight += w
            if scores:
                winner = max(scores, key=scores.get)
                conf = scores[winner] / total_weight if total_weight > 0 else 0
                final_plate = winner
                vote_info = f"Score {int(scores[winner])} ({conf:.1%})"

        final_plate, final_type = vehicle_classifier.resolve_type(
            record.get('class_id'), record.get('plate_history', [])
        )

        # 3. 物理统计 (速度 & 里程)
        dist_m = record.get('total_distance_m', 0.0)
        max_spd = record.get('max_speed', 0.0)
        
        avg_spd = dist_m / duration if duration > 0 else 0
        avg_spd_kmh = avg_spd * 3.6
        
        speed_info = f"Avg: {avg_spd_kmh:.1f} km/h | Max: {max_spd:.1f} m/s | Dist: {dist_m:.1f} m"
        
        # 打印报告头
        print("-" * 70)
        print(f"[Exit] ID: {tid} | Life: {duration:.1f}s | Type: {final_type}")
        print(f"       Plate: {final_plate} [{vote_info}]")
        print(f"       Physics: {speed_info}")
        
        # 4. 绘制运动学曲线 (速度 & 加速度)
        trajectory = record.get('trajectory', [])
        if len(trajectory) > 5:
            speeds = [p['speed'] for p in trajectory]
            accels = [p['accel'] for p in trajectory]
            print("\n       [Kinematics Profile]")
            self._plot_kinematics_graph(speeds, accels)
            
        print("-" * 70 + "\n")

    def _plot_kinematics_graph(self, speeds, accels):
        """绘制运动学曲线 (保持原有绘图逻辑不变)"""
        term_w, term_h = plt.terminal_size()
        safe_w = min(term_w - 5, 100) 
        safe_h = 31 
        
        if safe_w < 40 or term_h < 36: return 

        plt.clear_figure()
        plt.plotsize(safe_w, safe_h)
        plt.subplots(2, 1)
        
        t = [i / self.fps for i in range(len(speeds))]
        DENSE_TICKS_COUNT = 13

        # --- 子图 1: 速度曲线 ---
        plt.subplot(1, 1)
        plt.plot(t, speeds, marker="dot", color="cyan")
        plt.title("Speed (m/s)")
        plt.theme('dark')
        plt.ticks_color('white') 
        plt.grid(True, True)
        
        max_v = max(speeds) if speeds else 0
        limit_v = max(max_v * 1.05, 1.0) 
        plt.ylim(0, limit_v)
        plt.yticks(np.linspace(0, limit_v, DENSE_TICKS_COUNT).tolist()) 

        # --- 子图 2: 加速度曲线 ---
        plt.subplot(2, 1)
        plt.plot(t, accels, marker="dot", color="magenta")
        plt.title("Accel (m/s^2)")
        plt.theme('dark')
        plt.ticks_color('white')
        plt.grid(True, True)
        
        max_abs_a = max([abs(x) for x in accels]) if accels else 0
        limit_a = max(max_abs_a * 1.05, 0.5) 
        plt.ylim(-limit_a, limit_a)
        plt.yticks(np.linspace(-limit_a, limit_a, DENSE_TICKS_COUNT).tolist())
        plt.hline(0, color="red") 
        
        plt.show()
