# 文件路径：app/alignment_daemon.py
import time
import logging
from domain.physics.alignment_engine import DelayedAlignmentEngine
from infra.store.sqlite_manager import DatabaseManager
import infra.config.loader as cfg

logger = logging.getLogger(__name__)

class AlignmentDaemon:
    """
    独立于视觉流的后台守护进程。
    通过轮询 SQLite 获取最新的时空游标，驱动延迟对齐引擎。
    """
    def __init__(self, config, stop_event=None):
        self.config = config
        self.engine = DelayedAlignmentEngine(config, config.DB_PATH)
        self.stop_event = stop_event

    def run(self):
        logger.info("延迟对齐后台进程已启动，开始监听数据库...")
        db = DatabaseManager(self.config.DB_PATH)
        
        # 未收到停止信号前，循环执行
        while not (self.stop_event and self.stop_event.is_set()):
            try:
                # 1. 查找当前系统中最新启动的 Session
                db.cursor.execute("SELECT session_id FROM Session_Task ORDER BY start_time DESC LIMIT 1")
                row = db.cursor.fetchone()
                if not row:
                    time.sleep(2)
                    continue
                
                session_id = row[0]
                
                # 2. 核心解耦点：从环境表中获取最大的绝对时间戳。
                # 视觉引擎每秒都会写入 Env_Raw，这里的时间戳就代表了系统的“当前物理时间”
                db.cursor.execute("SELECT MAX(timestamp) FROM Env_Raw WHERE session_id = ?", (session_id,))
                t_row = db.cursor.fetchone()
                
                if t_row and t_row[0]:
                    latest_timestamp = t_row[0]
                    # 3. 触发对齐运算（引擎内部的 align_interval 会自动节流，防止过度运算）
                    self.engine.align_step(session_id, latest_timestamp)
                
            except Exception as e:
                # 可打印完整 Traceback，捕获如 SQLite 锁死等严重错误
                logger.exception("后台对齐调度发生严重异常.")
            
            # 基础轮询休眠，防止死循环打满 CPU
            time.sleep(1.0)
            
        db.close()
        logger.info("延迟对齐后台进程已安全退出.")

    def stop(self):
        self.is_running = False
