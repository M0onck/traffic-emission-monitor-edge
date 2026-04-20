# perception/daemon.py
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import logging
import gc
import os
import threading

logger = logging.getLogger(__name__)

def perception_worker(shm_name, shape, bbox_queue, stop_event, config_dict):
    """
    独立的系统级感知进程 (Hard Real-time 工业级版本)。
    拥有完全独立的 Python 解释器状态和 GC，绝不被主进程的 UI 渲染阻塞。
    """
    # ==========================================
    # 关闭自动全局垃圾回收
    # 彻底禁止 Python 在后台进行不可控的 Stop-The-World 停顿
    # ==========================================
    gc.disable()

    # [非常重要] 必须在子进程内部导入 GStreamer 相关库！
    # 否则 fork 进程时会引发 GObject 信号系统的底层崩溃
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    import infra.config.loader as config
    from perception.gst_pipeline import GstPipelineManager
    
    class ConfigWrapper:
        def __init__(self, d):
            for k, v in d.items(): setattr(self, k, v)

    runtime_config = ConfigWrapper(config_dict)

    pipeline = None
    existing_shm = None
    
    try:
        logger.info("-> [感知进程] 初始化 NPU 与流水线...")
        
        # 接入主进程开辟好的操作系统级共享内存
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        shm_array = np.ndarray(shape, dtype=np.uint8, buffer=existing_shm.buf)

        # 创建一个事件锁，作为两界的起搏器
        frame_ready_event = threading.Event()

        pipeline = GstPipelineManager(runtime_config, shm_array=shm_array, frame_ready_event=frame_ready_event)
        pipeline.start()
        logger.info("-> [感知进程] 回调模式已启动，正在监听信号...")

        while not stop_event.is_set():
            # 如果 0.2 秒没画面（说明硬件卡顿或已暂停），就进入下一轮循环检查 stop_event
            if frame_ready_event.wait(timeout=0.2):
                frame_ready_event.clear() # 收到信号，立即重置状态
                
                hailo_data = pipeline.read_metadata()
                
                # 极致泄压：清空积压，永远只给主进程推最新一帧的元数据
                while not bbox_queue.empty():
                    try: bbox_queue.get_nowait()
                    except: pass
                
                try: bbox_queue.put_nowait(hailo_data)
                except: pass

    except KeyboardInterrupt:
        logger.info("[感知进程] 收到键盘中断信号。")
    except Exception as e:
        logger.error(f"[感知进程] 发生致命错误: {e}", exc_info=True)
    finally:
        logger.info("-> [感知进程] 准备退出，正在安全拆除 GStreamer 管道...")
        
        # 退出前恢复系统默认设置
        gc.enable()
        
        if pipeline:
            pipeline.stop()
        if existing_shm:
            existing_shm.close()
        logger.info("-> [感知进程] 已彻底安全销毁，硬件资源释放完毕。")
