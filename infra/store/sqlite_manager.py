import sqlite3
import json
import os
import numpy as np
import re # 正则表达式
import threading
from typing import List, Dict, Any

class NumpyEncoder(json.JSONEncoder):
    """JSON 编码器，解决 Numpy 数据类型无法被 json.dumps 序列化的问题"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class DatabaseManager:
    """
    [基础层] SQLite 数据库管理器 (Final)
    功能：负责微观数据的批量写入和宏观数据的汇总存储。
    改进：完全剥离 SQL 语句（DDL 在 schema.sql, DML 在 queries.sql）。
    """
    def __init__(self, db_path: str = "data/database/recorded_data.db", fps: float = 30.0):
        # 自动提取父路径，并递归创建目录结构
        db_dir = os.path.dirname(db_path)
        if db_dir:  # 确保路径不为空
            os.makedirs(db_dir, exist_ok=True)
        self.db_path = db_path
        self.fps = fps

        # 初始化线程同步锁
        self.lock = threading.Lock()

        # 允许后台线程读写
        self.conn = sqlite3.connect(
            self.db_path, 
            timeout=10.0, 
            check_same_thread=False  
        )
        
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        
        # 1. 初始化表结构 (DDL)
        self._init_schema()
        
        # 2. 加载查询模板 (DML)
        self.queries = self._load_queries()

    def _init_schema(self):
        """加载并执行 schema.sql"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(current_dir, 'schema.sql')

        if os.path.exists(schema_path):
            try:
                with open(schema_path, 'r', encoding='utf-8') as f:
                    self.conn.executescript(f.read())
                self.conn.commit()
            except sqlite3.Error as e:
                print(f"[Database Error] Schema 初始化失败: {e}")
        else:
            print(f"[Database Warning] Schema 文件未找到: {schema_path}")

    def _load_queries(self) -> Dict[str, str]:
        """
        解析 queries.sql 文件
        返回格式: {'insert_micro_log': 'INSERT INTO...', ...}
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        query_path = os.path.join(current_dir, 'queries.sql')
        queries = {}

        if not os.path.exists(query_path):
            print(f"[Database Error] 查询文件未找到: {query_path}")
            return queries

        try:
            with open(query_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用正则匹配 "-- name: key" 块
            # 逻辑：匹配 name 行，捕获名称，然后非贪婪匹配后续内容直到下一个 name 行或文件结束
            pattern = re.compile(r'--\s*name:\s*(\w+)\s+(.+?)(?=--\s*name:|\Z)', re.DOTALL)
            
            for match in pattern.finditer(content):
                name = match.group(1).strip()
                sql = match.group(2).strip()
                # 移除行尾分号（sqlite3 execute 不需要分号，有时甚至会报错）
                if sql.endswith(';'):
                    sql = sql[:-1]
                queries[name] = sql
                
            print(f">>> [Database] 已加载 {len(queries)} 条 SQL 模板")
            
        except Exception as e:
            print(f"[Database Error] 解析 queries.sql 失败: {e}")
            
        return queries

    def create_session(self, session_id: str, start_time: float, location_desc: str = "默认路口"):
        """
        创建一个新的采集任务会话
        """
        sql = self.queries.get('insert_session')
        if sql:
            try:
                self.conn.execute(sql, (session_id, start_time, location_desc))
                self.conn.commit()
                print(f">>> [Database] 成功创建新采集任务: {session_id}")
            except Exception as e:
                print(f"[Database Error] 创建任务会话失败: {e}")
        else:
            print("[Database Error] SQL模板 'insert_session' 未定义")

    def complete_session(self, session_id: str, end_time: float):
        """
        结束当前的采集任务会话
        """
        sql = self.queries.get('complete_session')
        if sql:
            try:
                self.conn.execute(sql, (end_time, session_id))
                self.conn.commit()
                print(f">>> [Database] 采集任务已结束并归档: {session_id}")
            except Exception as e:
                print(f"[Database Error] 结束任务会话失败: {e}")
        else:
            print("[Database Error] SQL模板 'complete_session' 未定义")

    def fetch_all_sessions(self) -> List[tuple]:
        """查询所有历史采集任务，用于填充下拉菜单"""
        query = "SELECT session_id, start_time, location_desc FROM Session_Task ORDER BY start_time DESC"
        try:
            # 显式创建局部游标
            cursor = self.conn.cursor()
            cursor.execute(query)
            # 从游标中获取数据
            res = cursor.fetchall()
            # 及时关闭局部游标释放资源
            cursor.close()
            return res
        except sqlite3.Error as e:
            print(f"[Database Error] 查询 Session_Task 失败: {e}")
            return []

    def update_session_parameters(self, session_id: str, calibration_params_json: str, physical_priors_json: str):
        """
        将 UI 传来的标定和物理先验参数 (已转为JSON字符串) 与 session_id 绑定并落盘
        """
        query = """
            UPDATE Session_Task
            SET calibration_params = ?,
                physical_priors = ?
            WHERE session_id = ?
        """
        try:
            # 统一使用 self.conn 和线程锁，移除所有多余的 json.dumps
            with self.lock:
                self.conn.execute(
                    query,
                    (calibration_params_json, physical_priors_json, session_id)
                )
                self.conn.commit()
                print(f">>> [Database] 成功更新任务 {session_id} 的标定与物理先验参数")
        except Exception as e:
            # 移除不存在的 self.logger，统一用 print
            print(f"[Database Error] 更新 Session 参数失败: {e}")

    def fetch_macro_records_by_session(self, session_id: str, limit: int = 50) -> List[tuple]:
        """获取指定采集任务的车辆记录"""
        query = """
            SELECT tracker_id, vehicle_type, energy_type, 
                   entry_time, exit_time, ROUND(average_speed, 2), dominant_opmodes, settlement_status
            FROM Veh_Sum 
            WHERE session_id = ?
            ORDER BY exit_time DESC 
            LIMIT ?
        """
        try:
            # 显式创建局部游标
            cursor = self.conn.cursor()
            cursor.execute(query, (session_id, limit))
            # 从游标中获取数据
            res = cursor.fetchall()
            # 及时关闭局部游标
            cursor.close()
            return res
        except sqlite3.Error as e:
            print(f"[Database Error] 查询 Veh_Sum 数据失败: {e}")
            return []

    def delete_session(self, session_id: str) -> bool:
        """删除指定采集任务的所有关联数据"""
        try:
            self.conn.execute("DELETE FROM Env_Raw WHERE session_id = ?", (session_id,))
            self.conn.execute("DELETE FROM Veh_Raw WHERE session_id = ?", (session_id,))
            self.conn.execute("DELETE FROM Veh_Sum WHERE session_id = ?", (session_id,))
            self.conn.execute("DELETE FROM Aligned_Snapshots WHERE session_id = ?", (session_id,))
            self.conn.execute("DELETE FROM Session_Task WHERE session_id = ?", (session_id,))
            self.conn.commit()
            print(f">>> [Database] 任务 {session_id} 数据已被彻底删除。")
            return True
        except sqlite3.Error as e:
            print(f"[Database Error] 删除任务 {session_id} 失败: {e}")
            return False

    def delete_all_data(self) -> bool:
        """清空数据库中的所有任务数据"""
        try:
            self.conn.execute("DELETE FROM Env_Raw")
            self.conn.execute("DELETE FROM Veh_Raw")
            self.conn.execute("DELETE FROM Veh_Sum")
            self.conn.execute("DELETE FROM Aligned_Snapshots")
            self.conn.execute("DELETE FROM Session_Task")
            self.conn.commit()
            print(">>> [Database] 所有历史数据已被清空。")
            return True
        except sqlite3.Error as e:
            print(f"[Database Error] 清空数据失败: {e}")
            return False

    def insert_env_raw(self, session_id: str, timestamp: float, env_data: dict):
        """
        高频实时写入环境微观数据 (1Hz)
        """
        sql = self.queries.get('insert_env_raw')
        if sql:
            # 使用 .get() 附带默认兜底值，防止传感器离线导致报错
            params = (
                session_id,
                timestamp,
                float(env_data.get('pm25_raw', 0.0)),
                float(env_data.get('pm10_raw', 0.0)),
                float(env_data.get('wind_speed', 0.0)),
                float(env_data.get('wind_dir', 0.0)),
                float(env_data.get('air_temp', 0.0)),
                float(env_data.get('humidity', 0.0)),
                float(env_data.get('ground_temp', 0.0))
            )
            try:
                with self.lock:
                    self.conn.execute(sql, params)
                    self.conn.commit() # 1Hz的写入频率较低，由于开启了WAL模式，直接commit保证实时性无压力
            except Exception as e:
                print(f"[Database Error] 插入 Env_Raw 失败: {e}")
        else:
            print("[Database Error] SQL模板 'insert_env_raw' 未定义")

    def insert_veh_raw(self, session_id: str, tid: int, vehicle_type: str, energy_type: str, entry_time: float, exit_time: float, trajectory: list):
        """
        车辆离场后，写入一条包含完整平滑轨迹的微观记录
        """
        sql = self.queries.get('insert_veh_raw')
        if sql:
            try:
                # 将列表形式的物理轨迹序列化为 JSON 字符串
                trajectory_blob = json.dumps(trajectory, cls=NumpyEncoder)
                params = (
                    session_id,
                    int(tid),
                    str(vehicle_type),
                    str(energy_type),
                    float(entry_time),
                    float(exit_time),
                    trajectory_blob
                )
                with self.lock:
                    self.conn.execute(sql, params)
                    self.conn.commit() 
            except Exception as e:
                print(f"[Database Error] 插入 Veh_Raw 失败: {e}")
        else:
            print("[Database Error] SQL模板 'insert_veh_raw' 未定义")

    def insert_veh_sum(self, session_id: str, tid: int, record: dict, vehicle_type: str, energy_type: str, dominant_opmodes: list, settlement_status: str):
        """
        向前端看板专用表写入精简后的车辆统计数据
        """
        speed_count = record.get('speed_count', 0)
        avg_speed = record.get('speed_sum', 0.0) / speed_count if speed_count > 0 else 0.0
        opmodes_json = json.dumps(dominant_opmodes)

        # 构建与 queries.sql 对应的参数元组
        params = (
            session_id,
            int(tid),
            str(vehicle_type),
            str(energy_type),
            float(record.get('first_time', 0.0)),
            float(record.get('last_seen_time', 0.0)),
            float(avg_speed),
            opmodes_json,
            str(settlement_status)
        )
        
        sql = self.queries.get('insert_veh_sum')
        if sql:
            try:
                with self.lock:
                    self.conn.execute(sql, params)
                    self.conn.commit()
            except Exception as e:
                print(f"[Database Error] insert_veh_sum 失败: {e}")
        else:
            print("[Database Error] SQL模板 'insert_veh_sum' 未定义")

    def insert_aligned_snapshot(self, snapshot: dict):
        """
        写入对齐后的 L2 级物理切片快照
        """
        sql = self.queries.get('insert_aligned_snapshot')
        if sql:
            # 安全提取字典，空值写入 NULL (None)
            params = (
                snapshot['session_id'], 
                float(snapshot['aligned_timestamp']), 
                snapshot.get('air_temp'), 
                snapshot.get('ground_temp'), 
                snapshot.get('humidity'), 
                snapshot.get('wind_speed'), 
                snapshot.get('wind_dir'), 
                snapshot.get('pm25'), 
                snapshot.get('pm10'),
                int(snapshot['active_vehicle_count']),
                snapshot['vehicles_data'] # 直接存入 JSON 字符串
            )
            try:
                with self.lock:
                    self.conn.execute(sql, params)
                    self.conn.commit()
            except Exception as e:
                print(f"[Database Error] 插入 Aligned_Snapshots 失败: {e}")
        else:
            print("[Database Error] SQL模板 'insert_aligned_snapshot' 未定义")

    def get_table_data_for_export(self, table_name: str, session_id: str):
        """
        根据表名和 session_id 获取所有原始数据记录及表头，用于 CSV 导出
        """
        # 基本的白名单防御，防止 SQL 注入或意外读取系统表
        allowed_tables = ["Veh_Sum", "Veh_Raw", "Env_Raw", "Aligned_Snapshots", "Session_Task"]
        if table_name not in allowed_tables:
            print(f"[Database Error] 尝试导出的表名 {table_name} 不合法。")
            return None, None

        # 动态拼接表名 (白名单校验后拼接是安全的)
        query = f"SELECT * FROM {table_name} WHERE session_id = ?"
        try:
            # 显式创建局部游标，防止多线程跨进程读取时发生 Cursor 冲突
            cursor = self.conn.cursor()
            cursor.execute(query, (session_id,))
            
            # 从游标的 description 中提取所有列名作为 CSV 的表头
            columns = [column[0] for column in cursor.description]
            # 获取该 session_id 下的所有行数据
            rows = cursor.fetchall()
            
            # 及时关闭局部游标
            cursor.close()
            return columns, rows
            
        except sqlite3.Error as e:
            print(f"[Database Error] 导出数据时查询表 {table_name} 失败: {e}")
            return None, None

    def get_nearest_env_raw(self, session_id: str, target_time: float, max_tolerance: float) -> dict:
        """获取离目标时间戳最近的一条环境记录"""
        sql = self.queries.get('get_nearest_env_raw')
        if not sql: return {}
        try:
            cursor = self.conn.cursor()
            # 对应 SQL 中的 4 个 '?' : session_id, target_time, max_tolerance, target_time
            cursor.execute(sql, (session_id, target_time, max_tolerance, target_time))
            row = cursor.fetchone()
            if row:
                cols = [column[0] for column in cursor.description]
                result = dict(zip(cols, row))
                cursor.close()
                return result
            cursor.close()
            return {}
        except sqlite3.Error as e:
            print(f"[Database Error] get_nearest_env_raw 失败: {e}")
            return {}

    def get_active_vehicles_during(self, session_id: str, target_time: float) -> list:
        """获取在特定时间点处于场内的所有车辆及其原始轨迹 BLOB"""
        sql = self.queries.get('get_active_vehicles_during')
        if not sql: return []
        try:
            cursor = self.conn.cursor()
            # 对应 SQL 中的 3 个 '?' : session_id, target_time, target_time
            cursor.execute(sql, (session_id, target_time, target_time))
            rows = cursor.fetchall()
            cols = [column[0] for column in cursor.description]
            cursor.close()
            return [dict(zip(cols, row)) for row in rows]
        except sqlite3.Error as e:
            print(f"[Database Error] get_active_vehicles_during 失败: {e}")
            return []

    def close(self):
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            self.conn.close()
            print("[Database] 数据库连接已安全关闭，WAL 缓存已全部落盘。")
        except Exception as e:
            print(f"[Database Error] 关闭数据库时发生异常: {e}")
