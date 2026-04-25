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
        self.db = DatabaseManager(config.DB_PATH, getattr(config, 'FPS', 30.0))
        self.engine = AlignmentEngine(self.db)
        
        # 记录上一次成功对齐的时间戳，用于断层补齐
        self.last_aligned_time = None 
        self.current_session = None

    def run(self):
        logger.info("[AlignmentDaemon] 延迟对齐守护进程已启动...")
        
        try:
            while not (self.stop_event and self.stop_event.is_set()):
                try:
                    session_id, latest_timestamp = self.sync_queue.get(timeout=1.0)
                    self.current_session = session_id
                    
                    # 初始化基准时间
                    if self.last_aligned_time is None:
                        self.last_aligned_time = latest_timestamp - 1.0

                    # 如果由于看门狗强杀导致主线程卡顿，latest_timestamp 突然跳跃了 12 秒
                    # 这个 while 循环会精准地逐秒推进，把漏掉的 12 秒快照全部补齐
                    while self.last_aligned_time < latest_timestamp:
                        target_tick = self.last_aligned_time + 1.0
                        
                        # 防止由于浮点精度问题超调
                        if target_tick > latest_timestamp:
                            break
                            
                        snapshot = self.engine.process_alignment_tick(session_id, target_tick)
                        if snapshot:
                            self.db.insert_aligned_snapshot(snapshot)
                            if getattr(self.config, 'MODE', 'collect') == 'inference':
                                pass # TODO 送入预测模型
                                
                        self.last_aligned_time = target_tick
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.exception("[AlignmentDaemon] 守护进程内部发生异常.")
                    
        finally:
            # 停机后资源回收
            if self.current_session:
                logger.info(f"[AlignmentDaemon] 收到停机指令。已舍弃末尾 {self.config.ALIGNMENT_DELAY_SEC} 秒的未决数据。")
                
            self.db.close()
            logger.info("[AlignmentDaemon] 数据库连接已释放，守护进程安全退出.")
