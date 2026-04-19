# perception/daemon.py
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

def perception_worker(shm_name, shape, bbox_queue, stop_event, config_dict):
    """
    独立的系统级感知进程。
    拥有完全独立的 Python 解释器状态和 GC，绝不被主进程的 UI 渲染阻塞。
    """
    # [非常重要] 必须在子进程内部导入 GStreamer 相关库！
    # 否则 fork 进程时会引发 GObject 信号系统的底层崩溃
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
        pipeline = GstPipelineManager(runtime_config)
        pipeline.start()
        
        # 接入主进程开辟好的操作系统级共享内存
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        shm_array = np.ndarray(shape, dtype=np.uint8, buffer=existing_shm.buf)
        logger.info(f"-> [感知进程] 共享内存 '{shm_name}' 挂载成功，开始极速推流！")

        while not stop_event.is_set():
            frame, hailo_data = pipeline.read()
            
            if frame is not None:
                # 1. 画面数据：零拷贝瞬间覆写共享内存 (约 0.1 毫秒)
                np.copyto(shm_array, frame)
                
                # 2. AI 数据：非阻塞覆写队列（极致泄压逻辑）
                # 只要主进程没取走，我们直接把旧数据扔掉，永远只塞最新的一帧
                while not bbox_queue.empty():
                    try:
                        bbox_queue.get_nowait()
                    except:
                        pass
                        
                try:
                    bbox_queue.put_nowait(hailo_data)
                except:
                    pass
            else:
                # 管道若暂时无数据（如 GStreamer 内部等待），给 CPU 1毫秒喘息时间
                time.sleep(0.001)

    except KeyboardInterrupt:
        logger.info("[感知进程] 收到键盘中断信号。")
    except Exception as e:
        logger.error(f"[感知进程] 发生致命错误: {e}", exc_info=True)
    finally:
        logger.info("-> [感知进程] 准备退出，正在安全拆除 GStreamer 管道...")
        if pipeline:
            pipeline.stop()
        if existing_shm:
            existing_shm.close()
        logger.info("-> [感知进程] 已彻底安全销毁，硬件资源释放完毕。")
