import os
import csv
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class StorageManager:
    # 容器内固定挂载点
    DATA_ROOT = Path("data")
    USB_ROOT = Path("/media")
    
    # 子功能目录
    DB_DIR = DATA_ROOT / "database"
    REC_DIR = DATA_ROOT / "recorded_videos"
    TEST_DIR = DATA_ROOT / "test_videos"

    @classmethod
    def ensure_structure(cls):
        """初始化必要的文件夹结构"""
        for p in [cls.DB_DIR, cls.REC_DIR, cls.TEST_DIR]:
            p.mkdir(parents=True, exist_ok=True)

    @classmethod
    def list_test_videos(cls):
        """只列出内部测试文件夹中的视频文件"""
        extensions = ('.mp4', '.avi', '.mkv', '.mov')
        return [f.name for f in cls.TEST_DIR.iterdir() if f.suffix.lower() in extensions]

    @classmethod
    def get_available_usbs(cls):
        """获取当前挂载的所有 U 盘目录"""
        if not cls.USB_ROOT.exists():
            return []
            
        usbs = []
        try:
            for item in cls.USB_ROOT.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    try:
                        # 尝试向下探索一层 (应对 /media/username/USB_NAME 的情况)
                        sub_dirs = [d for d in item.iterdir() if d.is_dir() and not d.name.startswith('.')]
                        if sub_dirs:
                            usbs.extend(sub_dirs)
                        else:
                            usbs.append(item)
                    except PermissionError:
                        # 遇到属于其他用户或系统锁定的私有目录，直接无视并跳过
                        continue
        except PermissionError:
            # 如果连 /media 本身的读取权限都没有
            return []

        # 对收集到的所有疑似路径进行真实 [写权限] 校验
        # 彻底过滤掉挂载的只读光驱、系统恢复盘等假目标
        valid_usbs = [usb for usb in usbs if os.access(usb, os.W_OK)]
        
        return valid_usbs

    @classmethod
    def import_from_usb(cls, usb_path, file_name):
        """将测试视频从 U 盘导入内置 SSD"""
        src = Path(usb_path) / file_name
        dest = cls.TEST_DIR / file_name
        shutil.copy2(src, dest)
        return dest

    @classmethod
    def export_to_usb(cls, video_name, usb_target_path, session_id=None):
        """将录制的视频导出到 U 盘，并支持建立独立的 Session 文件夹"""
        src = cls.REC_DIR / video_name
        target_base = Path(usb_target_path)
        
        # 如果提供了 session_id，则在 U 盘根目录新建专属文件夹
        if session_id:
            target_base = target_base / f"{session_id}_recorded_videos"
            target_base.mkdir(parents=True, exist_ok=True) # 递归创建并赋予读写权限
            
        dest = target_base / video_name
        shutil.copy2(src, dest)
        return dest

    @classmethod
    def get_session_videos(cls) -> dict:
        """扫描录像目录，按 session_id 归类视频文件"""
        if not cls.REC_DIR.exists():
            return {}
            
        session_map = {}
        for file in cls.REC_DIR.iterdir():
            if file.suffix.lower() in ['.mp4', '.mkv'] and '_seq' in file.name:
                # 提取 session_id (前缀部分)
                session_id = file.name.split('_seq')[0]
                if session_id not in session_map:
                    session_map[session_id] = []
                session_map[session_id].append(file.name)
                
        return session_map

    def get_table_data_for_export(self, table_name, session_id):
        """根据表名和 session_id 获取所有原始数据记录"""
        query = f"SELECT * FROM {table_name} WHERE session_id = ?"
        try:
            self.cursor.execute(query, (session_id,))
            # 获取表头字段名
            columns = [column[0] for column in self.cursor.description]
            # 获取所有行数据
            rows = self.cursor.fetchall()
            return columns, rows
        except Exception as e:
            logger.error(f"导出数据时查询表 {table_name} 失败: {e}")
            return None, None
        
    @classmethod
    def export_data_to_usb(cls, session_id, usb_target_path, db_manager):
        """将指定 Session 的所有数据库表导出为 CSV 文件至 U 盘"""
        # 1. 创建 session_id + "data" 命名的文件夹
        target_dir = Path(usb_target_path) / f"{session_id}_data"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. 定义需要导出的核心业务表
        tables_to_export = ["Veh_Sum", "Veh_Raw", "Env_Raw"]
        exported_files = []

        for table in tables_to_export:
            cols, rows = db_manager.get_table_data_for_export(table, session_id)
            if not cols or not rows:
                continue
                
            csv_path = target_dir / f"{table}.csv"
            # 使用 utf-8-sig 编码以确保导出的 CSV 在 Excel 中打开不会乱码
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(cols) # 写入表头
                writer.writerows(rows) # 写入数据行
            exported_files.append(csv_path.name)
            
        return target_dir, exported_files
