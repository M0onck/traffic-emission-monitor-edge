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
        self._migrate_old_data() # 清洗历史数据
        
        # 2. 加载查询模板 (DML)
        self.queries = self._load_queries()
        
        self.micro_buffer: List[tuple] = []
        self.BATCH_SIZE = 100 

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

    def _migrate_old_data(self):
        """
        [数据迁移] 将历史遗留的 YOLO 英文类别强制清洗为排放大类
        旧数据：car, bus, truck (或大写)
        新数据映射：LDV-Gasoline, HDV-Diesel
        """
        try:
            # 1. 将 bus 和 truck 替换为重型柴油车 (HDV-Diesel)
            self.cursor.execute("""
                UPDATE vehicle_macro 
                SET class_name = 'HDV-Diesel' 
                WHERE class_name IN ('bus', 'truck', 'Bus', 'Truck')
            """)
            
            # 2. 将 car 替换为轻型燃油车 (LDV-Gasoline)
            self.cursor.execute("""
                UPDATE vehicle_macro 
                SET class_name = 'LDV-Gasoline' 
                WHERE class_name IN ('car', 'Car')
            """)
            
            self.conn.commit()
            print(">>> [Database] 历史数据校验与清洗完成 (Car->LDV, Bus/Truck->HDV)")
        except sqlite3.Error as e:
            print(f"[Database Error] 历史数据迁移失败: {e}")

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

    def insert_micro(self, fid: int, tid: int, payload: dict):
        # 只提取最核心的 3 个物理维度数据，极大地降低序列化与 I/O 开销
        params = (
            int(tid),
            int(fid),
            float(payload.get('timestamp', 0.0)),
            float(payload.get('ipm_x', 0.0)),
            float(payload.get('ipm_y', 0.0))
        )
        self.micro_buffer.append(params)
        if len(self.micro_buffer) >= self.BATCH_SIZE:
            self.flush_micro_buffer()

    def flush_micro_buffer(self):
        """强制写入微观数据缓冲区"""
        if not self.micro_buffer: return
        
        sql = self.queries.get('insert_micro')
        if not sql:
            print("[Database Error] SQL模板 'insert_micro' 未定义")
            return
            
        try:
            self.cursor.executemany(sql, self.micro_buffer)
            self.conn.commit()
            self.micro_buffer.clear()
        except Exception as e:
            print(f"[Database Error] Batch insert failed: {e}")

    def insert_macro(self, tid: int, record: dict, vehicle_type: str, energy_type: str, dominant_opmodes: list):
        """
        向宏观表写入精简后的车辆统计数据
        :param dominant_opmodes: list，例如 ["Cruise", "Braking"]
        """
        # 计算平均速度
        speed_count = record.get('speed_count', 0)
        avg_speed = record.get('speed_sum', 0.0) / speed_count if speed_count > 0 else 0.0
        
        # 将工况数组序列化为 JSON 字符串
        opmodes_json = json.dumps(dominant_opmodes)

        # 构建与 queries.sql 对应的参数元组
        params = (
            int(tid),
            str(vehicle_type),
            str(energy_type),
            float(record.get('first_time', 0.0)),     # 依赖 repository.py 中的 first_time
            float(record.get('last_seen_time', 0.0)), # 依赖 repository.py 中的 last_seen_time
            float(avg_speed),
            opmodes_json
        )
        
        sql = self.queries.get('insert_macro')
        if sql:
            try:
                self.cursor.execute(sql, params)
                self.conn.commit()
            except Exception as e:
                print(f"[Database Error] insert_macro 失败: {e}")
        else:
            print("[Database Error] SQL模板 'insert_macro' 未定义")

    def fetch_recent_macro_records(self, limit: int = 50) -> List[tuple]:
        """
        获取最近写入的宏观车辆记录，限制条数以保护边缘设备内存
        """
        # 强制将内存中还没写入的缓冲刷入磁盘，保证查到最新数据
        self.flush_micro_buffer() 
        
        # 宏观表查询语句
        query = """
            SELECT tracker_id, vehicle_type, energy_type, 
                   entry_time, exit_time, ROUND(average_speed, 2), dominant_opmodes
            FROM vehicle_macro 
            ORDER BY tracker_id DESC 
            LIMIT ?
        """
        try:
            self.cursor.execute(query, (limit,))
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"[Database Error] 查询宏观数据失败: {e}")
            return []

    def delete_all_data(self) -> bool:
        """清空数据库中的所有宏观和微观记录"""
        try:
            self.cursor.execute("DELETE FROM vehicle_micro")
            self.cursor.execute("DELETE FROM vehicle_macro")
            self.conn.commit()
            print(">>> [Database] 所有历史数据已被清空。")
            return True
        except sqlite3.Error as e:
            print(f"[Database Error] 清空所有数据失败: {e}")
            return False

    def delete_recent_data(self, minutes: int) -> bool:
        """删除最近 N 分钟内入场的记录及其微观轨迹"""
        import time
        # 计算时间阈值
        threshold_time = time.time() - (minutes * 60)
        try:
            # 1. 级联删除微观表 (先删外键引用的子表)
            self.cursor.execute("""
                DELETE FROM vehicle_micro 
                WHERE tracker_id IN (
                    SELECT tracker_id FROM vehicle_macro WHERE entry_time >= ?
                )
            """, (threshold_time,))
            
            # 2. 删除宏观表 (再删主表)
            self.cursor.execute("DELETE FROM vehicle_macro WHERE entry_time >= ?", (threshold_time,))
            
            self.conn.commit()
            print(f">>> [Database] 已删除最近 {minutes} 分钟产生的数据。")
            return True
        except sqlite3.Error as e:
            print(f"[Database Error] 按时间段删除失败: {e}")
            return False

    def close(self):
        self.flush_micro_buffer()
        self.conn.close()
        print("[Database] Connection closed.")
