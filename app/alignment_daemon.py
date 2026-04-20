# 文件路径：app/alignment_daemon.py
import time
import logging
import queue
from domain.physics.alignment_engine import DelayedAlignmentEngine
from infra.store.sqlite_manager import DatabaseManager

logger = logging.getLogger(__name__)

class AlignmentDaemon:
    """
    独立于视觉流的后台守护进程。
    通过轮询 SQLite 获取最新的时空游标，驱动延迟对齐引擎。
    """
    def __init__(self, config, sync_queue, stop_event=None):
        self.config = config
        self.engine = DelayedAlignmentEngine(config, config.DB_PATH)
        self.sync_queue = sync_queue
        self.stop_event = stop_event

    def run(self):
        logger.info("[AlignmentDaemon] 延迟对齐后台进程已启动，开始监听内存队列...")
        
        while not (self.stop_event and self.stop_event.is_set()):
            try:
                # 阻塞等待主引擎推送时间戳，timeout 防死锁
                # 这里的阻塞不消耗 CPU，比 time.sleep 优雅
                session_id, latest_timestamp = self.sync_queue.get(timeout=2.0)
                
                # 收到主引擎的信号，直接执行对齐运算
                self.engine.align_step(session_id, latest_timestamp)
                
            except queue.Empty:
                # 2秒内主引擎没发新数据，安静地继续下一轮监听即可
                continue
            except Exception as e:
                logger.exception("[AlignmentDaemon] 后台对齐调度发生异常.")
                
        logger.info("[AlignmentDaemon] 延迟对齐后台进程已安全退出.")

    def stop(self):
        self.is_running = False
