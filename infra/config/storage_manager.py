import os
import shutil
from pathlib import Path

class StorageManager:
    # 容器内固定挂载点
    DATA_ROOT = Path("/app/data")
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
        # 排除隐藏目录和空挂载点
        return [d for d in cls.USB_ROOT.iterdir() if d.is_dir() and not d.name.startswith('.')]

    @classmethod
    def import_from_usb(cls, usb_path, file_name):
        """将测试视频从 U 盘导入内置 SSD"""
        src = Path(usb_path) / file_name
        dest = cls.TEST_DIR / file_name
        shutil.copy2(src, dest)
        return dest

    @classmethod
    def export_to_usb(cls, video_name, usb_target_path):
        """将录制的视频导出到 U 盘"""
        src = cls.REC_DIR / video_name
        dest = Path(usb_target_path) / video_name
        shutil.copy2(src, dest)
        return dest
