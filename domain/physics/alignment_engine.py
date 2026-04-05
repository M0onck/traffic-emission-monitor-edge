# 文件路径: domain/physics/alignment_engine.py
import json
import math
import sqlite3
import numpy as np
from infra.store.sqlite_manager import DatabaseManager

class DelayedAlignmentEngine:
    """
    [业务层] 延迟对齐与特征重构引擎
    职责: 基于滞后时间基准 (T_align)，定期扫描数据库，计算宏观时间窗内的扬尘累积通量、
          交通做功能量积分以及微气象特征，并将深度对齐后的结果持久化到 Aligned_Dataset 表。
    """
    def __init__(self, config: dict, db_path: str):
        self.db_path = db_path
        
        # 1. 解析时间窗口参数
        tw = config.get("time_windows", {})
        self.delay_sec = tw.get("alignment_delay_sec", 60.0)
        self.int_win_sec = tw.get("integration_window_sec", 30.0)
        self.base_win_min = tw.get("baseline_window_minute", 10.0)

        # 2. 解析物理先验参数
        pp = config.get("physics_priors", {})
        self.wx_pos = pp.get("weather_station_x_pos", 0.0)
        self.road_dir = pp.get("road_direction_angle", 0.0)
        self.nev_penalty = pp.get("nev_mass_penalty_ratio", 1.2)

        # 基础质量先验 (吨) - 用于从运动学参数重构绝对动能
        self.mass_priors = {"LDV": 1.5, "HDV": 15.0} 
        self.last_align_time = 0.0

    def align_step(self, session_id: str, current_timestamp: float):
        """执行单步延迟对齐作业"""
        t_align = current_timestamp - self.delay_sec

        # 避免在同一秒内高频重复触发 (降频保护)
        if t_align - self.last_align_time < 0.9:
            return
        self.last_align_time = t_align

        # 定义积分窗口 [t_start, t_align] 和基线窗口 [t_base_start, t_align]
        t_start = t_align - self.int_win_sec
        t_base_start = t_align - (self.base_win_min * 60.0)

        # 在线程内独立实例化 DatabaseManager，保障多线程并发读取的 SQLite 安全性
        db = DatabaseManager(self.db_path)
        try:
            # =======================================================
            # 1. 获取目标变量与本底 (对应论文公式 1 和公式 2)
            # =======================================================
            # 1.1 计算动态本底 (过去 baseline_window_minute 内的粗颗粒物极小值)
            db.cursor.execute(
                "SELECT MIN(pm10_raw - pm25_raw) FROM Env_Raw WHERE session_id = ? AND timestamp BETWEEN ? AND ?",
                (session_id, t_base_start, t_align)
            )
            row = db.cursor.fetchone()
            pmc_baseline = row[0] if row and row[0] is not None else 0.0

            # 1.2 获取当前积分窗口内的气象时序序列
            db.cursor.execute(
                "SELECT timestamp, pm25_raw, pm10_raw, wind_speed, wind_dir, air_temp, humidity, ground_temp "
                "FROM Env_Raw WHERE session_id = ? AND timestamp BETWEEN ? AND ? ORDER BY timestamp ASC",
                (session_id, t_start, t_align)
            )
            env_rows = db.cursor.fetchall()
            
            if not env_rows:
                return # 窗口内缺失气象数据，无法对齐

            # 取 T_align 时刻的最末态微气象特征
            latest_env = env_rows[-1]
            pmc_raw = latest_env[2] - latest_env[1]

            # 1.3 积分计算扬尘累积通量 (Delta_C_flux)
            delta_c_flux = 0.0
            for i in range(1, len(env_rows)):
                dt = env_rows[i][0] - env_rows[i-1][0]
                pmc_i = env_rows[i][2] - env_rows[i][1]
                # 对有效增量进行黎曼积分： \int max(1.0, PMC - Base) dt
                delta_c_flux += max(1.0, pmc_i - pmc_baseline) * dt

            # =======================================================
            # 2. 核心干预变量重构 (交通做功) (对应论文公式 4 和公式 5)
            # =======================================================
            db.cursor.execute(
                "SELECT vehicle_type, energy_type, trajectory_blob "
                "FROM Veh_Raw WHERE session_id = ? AND exit_time >= ? AND entry_time <= ?",
                (session_id, t_start, t_align) # 提取时间窗内交集的车辆
            )
            veh_rows = db.cursor.fetchall()

            e_traffic = 0.0
            sum_d_times_e = 0.0 # 用于计算加权距离的分子

            for v_type, e_type, blob in veh_rows:
                # 质量分配与新能源修正
                base_type = v_type.split('-')[0] if v_type else "LDV"
                m_i = self.mass_priors.get(base_type, 1.5)
                if e_type == "Electric":
                    m_i *= self.nev_penalty

                try:
                    traj = json.loads(blob)
                except Exception:
                    continue
                
                e_i_vehicle = 0.0
                x_sum, pt_count = 0.0, 0
                
                # 在时间窗内对单车高频VSP进行数值积分
                traj.sort(key=lambda p: p.get('timestamp', 0))
                for i in range(1, len(traj)):
                    p_prev = traj[i-1]
                    p = traj[i]
                    t_p = p.get('timestamp', 0.0)
                    
                    if t_start <= t_p <= t_align:
                        dt = t_p - p_prev.get('timestamp', 0.0)
                        if 0 < dt < 0.5: # 剔除由于追踪中断造成的异常大跨步时间
                            vsp = p.get('vsp', 0.0)
                            e_point = vsp * m_i * dt
                            e_i_vehicle += e_point
                            x_sum += p.get('x', 0.0)
                            pt_count += 1
                            
                if e_i_vehicle > 0 and pt_count > 0:
                    e_traffic += e_i_vehicle
                    avg_x = x_sum / pt_count
                    d_i = abs(avg_x - self.wx_pos) # 单车与气象站的物理距离
                    sum_d_times_e += (d_i * e_i_vehicle)

            # 等效传输距离加权 (D_trans)
            d_trans = (sum_d_times_e / e_traffic) if e_traffic > 0 else 0.0

            # =======================================================
            # 3. 空间与环境协变量构建 (对应论文公式 6 和公式 7)
            # =======================================================
            w_spd, w_dir = latest_env[3], latest_env[4]
            t_air, rh, t_road = latest_env[5], latest_env[6], latest_env[7]

            # 3.1 有效横风扰动
            w_cross = w_spd * math.sin(math.radians(w_dir - self.road_dir))

            # 3.2 热力学虚温差 (带比湿推导)
            e_s = 6.112 * math.exp(17.67 * t_air / (t_air + 243.5)) # 饱和水汽压
            e_actual = (rh / 100.0) * e_s                           # 实际水汽压
            q = (0.622 * e_actual) / (1013.25 - 0.378 * e_actual)   # 比湿 (kg/kg)
            delta_tv = (t_road - t_air) * (1 + 0.61 * q)            # 虚温差

            # =======================================================
            # 4. 落盘持久化
            # =======================================================
            db.insert_aligned_dataset(
                session_id=session_id,
                aligned_timestamp=t_align,
                pmc_raw=pmc_raw,
                pmc_baseline=pmc_baseline,
                delta_c_flux=delta_c_flux,
                e_traffic=e_traffic,
                d_trans=d_trans,
                w_cross=w_cross,
                delta_tv=delta_tv
            )
            # print(f"[Alignment Engine] T_align {t_align:.1f} 数据簇提取对齐完毕.")

        except Exception as e:
            print(f"[AlignmentEngine Error] 对齐过程发生异常: {e}")
        finally:
            db.close()
