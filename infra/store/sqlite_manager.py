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

    def insert_macro(self, tid: int, record: dict, final_type_str: str, final_plate: str):
        # 计算平均速度
        speed_count = record.get('speed_count', 0)
        avg_speed = record.get('speed_sum', 0.0) / speed_count if speed_count > 0 else 0.0
        
        # 直接读取引擎层结算好的最终投票颜色
        plate_color = record.get('final_plate_color', 'Unknown')

        # 全面类型强转，并修复 final_type_str 未被使用的逻辑 Bug
        params = (
            int(tid),                                    # INTEGER PRIMARY KEY 必须是纯正的 int
            int(record.get('first_frame', 0)),
            int(record.get('last_seen_frame', 0)),
            int(record.get('class_id', -1)),
            str(final_type_str),                         # 使用传入的准确车型，替换掉之前的 get('class_name')
            str(final_plate),
            str(plate_color),
            float(record.get('max_speed', 0.0)),
            float(avg_speed),
            float(record.get('total_distance_m', 0.0))
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
        
        query = """
            SELECT tid, class_name, plate_number, plate_color, 
                ROUND(max_speed, 2), ROUND(avg_speed, 2), ROUND(total_distance, 2)
            FROM vehicle_macro 
            ORDER BY tid DESC 
            LIMIT ?
        """
        try:
            self.cursor.execute(query, (limit,))
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"[Database Error] 查询宏观数据失败: {e}")
            return []

    def close(self):
        self.flush_micro_buffer()
        self.conn.close()
        print("[Database] Connection closed.")
