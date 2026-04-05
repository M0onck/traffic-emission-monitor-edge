import sqlite3
import json
import os
import re # 正则表达式
from typing import List, Dict, Any

class DatabaseManager:
    """
    [基础层] SQLite 数据库管理器 (Final)
    功能：负责微观数据的批量写入和宏观数据的汇总存储。
    改进：完全剥离 SQL 语句（DDL 在 schema.sql, DML 在 queries.sql）。
    """
    def __init__(self, db_path: str = "data/traffic_data.db", fps: float = 30.0):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.fps = fps
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        self.cursor.execute("PRAGMA journal_mode=WAL;")
        self.cursor.execute("PRAGMA synchronous=NORMAL;")
        
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
                    self.cursor.executescript(f.read())
                self.conn.commit()
            except sqlite3.Error as e:
                print(f"[Database Error] Schema 初始化失败: {e}")
        else:
            print(f"[Database Warning] Schema 文件未找到: {schema_path}")

    def _load_queries(self) -> Dict[str, str]:
        """
        [核心逻辑] 解析 queries.sql 文件
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
                self.cursor.execute(sql, (session_id, start_time, location_desc))
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
                self.cursor.execute(sql, (end_time, session_id))
                self.conn.commit()
                print(f">>> [Database] 采集任务已结束并归档: {session_id}")
            except Exception as e:
                print(f"[Database Error] 结束任务会话失败: {e}")
        else:
            print("[Database Error] SQL模板 'complete_session' 未定义")

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
                self.cursor.execute(sql, params)
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
                trajectory_blob = json.dumps(trajectory)
                params = (
                    session_id,
                    int(tid),
                    str(vehicle_type),
                    str(energy_type),
                    float(entry_time),
                    float(exit_time),
                    trajectory_blob
                )
                self.cursor.execute(sql, params)
                self.conn.commit() 
            except Exception as e:
                print(f"[Database Error] 插入 Veh_Raw 失败: {e}")
        else:
            print("[Database Error] SQL模板 'insert_veh_raw' 未定义")

    def insert_veh_sum(self, session_id: str, tid: int, record: dict, vehicle_type: str, energy_type: str, dominant_opmodes: list):
        """
        向前端看板专用表写入精简后的车辆统计数据
        """
        speed_count = record.get('speed_count', 0)
        avg_speed = record.get('speed_sum', 0.0) / speed_count if speed_count > 0 else 0.0
        opmodes_json = json.dumps(dominant_opmodes)

        # 构建与 queries.sql 对应的参数元组 (新增了 session_id)
        params = (
            session_id,
            int(tid),
            str(vehicle_type),
            str(energy_type),
            float(record.get('first_time', 0.0)),
            float(record.get('last_seen_time', 0.0)),
            float(avg_speed),
            opmodes_json
        )
        
        sql = self.queries.get('insert_veh_sum')
        if sql:
            try:
                self.cursor.execute(sql, params)
                self.conn.commit()
            except Exception as e:
                print(f"[Database Error] insert_veh_sum 失败: {e}")
        else:
            print("[Database Error] SQL模板 'insert_veh_sum' 未定义")

    def insert_aligned_dataset(self, session_id: str, aligned_timestamp: float, 
                               pmc_raw: float, pmc_baseline: float, delta_c_flux: float, 
                               e_traffic: float, d_trans: float, w_cross: float, delta_tv: float):
        """
        写入对齐后的多源数据集，用于后续气象归一化反演
        """
        sql = self.queries.get('insert_aligned_dataset')
        if sql:
            params = (session_id, float(aligned_timestamp), 
                      float(pmc_raw), float(pmc_baseline), float(delta_c_flux), 
                      float(e_traffic), float(d_trans), float(w_cross), float(delta_tv))
            try:
                self.cursor.execute(sql, params)
                self.conn.commit()
            except Exception as e:
                print(f"[Database Error] 插入 Aligned_Dataset 失败: {e}")
        else:
            print("[Database Error] SQL模板 'insert_aligned_dataset' 未定义")

    def fetch_recent_macro_records(self, limit: int = 50) -> List[tuple]:
        """
        获取最近写入的宏观车辆记录，供前端 UI 表格渲染
        """
        query = """
            SELECT tracker_id, vehicle_type, energy_type, 
                   entry_time, exit_time, ROUND(average_speed, 2), dominant_opmodes
            FROM Veh_Sum 
            ORDER BY exit_time DESC 
            LIMIT ?
        """
        try:
            self.cursor.execute(query, (limit,))
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"[Database Error] 查询 Veh_Sum 数据失败: {e}")
            return []

    def close(self):
        self.flush_micro_buffer()
        self.conn.close()
        print("[Database] Connection closed.")
