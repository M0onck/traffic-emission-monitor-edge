# app/alignment_daemon.py
import time
import logging
import queue
from domain.physics.alignment_engine import AlignmentEngine
from infra.store.sqlite_manager import DatabaseManager

logger = logging.getLogger(__name__)

class AlignmentDaemon:
    """
    [后台服务] 数据对齐守护进程 (全模式兼容版)
    职责: 监听主时钟，驱动对齐引擎生成 1Hz 快照并无差别入库。
          如果系统处于推理模式，则将快照进一步推流给预测模型。
    """
    def __init__(self, config, sync_queue, stop_event=None):
        self.config = config
        self.sync_queue = sync_queue
        self.stop_event = stop_event
        
        # 为守护进程独立实例化一个 DB Manager。
        # 这样能彻底避免与主视界 UI / 感知进程争抢同一个 SQLite 内存游标，提高并发安全性
        self.db = DatabaseManager(config.DB_PATH, getattr(config, 'FPS', 30.0))
        
        # 实例化改版后的轻量级对齐引擎
        self.engine = AlignmentEngine(self.db)

    def run(self):
        logger.info("[AlignmentDaemon] 延迟对齐守护进程已启动，开始生成物理时空快照...")
        
        while not (self.stop_event and self.stop_event.is_set()):
            try:
                # 阻塞等待主引擎推送时间戳信号 (timeout 保证退出时能响应 stop_event)
                session_id, latest_timestamp = self.sync_queue.get(timeout=2.0)
                
                # 1. 核心动作：生成对齐快照
                snapshot = self.engine.process_alignment_tick(session_id, latest_timestamp)
                
                if snapshot:
                    # 2. 无差别持久化：所有模式下，都将 1Hz 纯净数据存入数据库
                    self.db.insert_aligned_snapshot(snapshot)
                    
                    # 3. 业务分流：仅在“推理模式”下，将数据喂给后续的模型流
                    if getattr(self.config, 'MODE', 'collect') == 'inference':
                        # 在此处接入未来的预测模型管道，例如:
                        # model_predictor.predict(snapshot)
                        pass
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception("[AlignmentDaemon] 守护进程内部发生异常.")
                
        # 安全退出前清理独立连接
        self.db.close()
        logger.info("[AlignmentDaemon] 延迟对齐守护进程已安全退出.")
