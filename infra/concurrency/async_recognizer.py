import threading
import queue
import logging
import numpy as np
import sys
import os

# ==========================================
# 1. 动态加载我们编译好的 C++ 模块
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# 推导 cpp_extension/build 的绝对路径
cpp_build_dir = os.path.abspath(os.path.join(current_dir, '../../perception/plate_classifier/cpp_extension/build'))
if cpp_build_dir not in sys.path:
    sys.path.append(cpp_build_dir)

try:
    import hailo_ocr_cpp
except ImportError as e:
    logging.error(f"无法加载 C++ OCR 扩展，请检查编译环境: {e}")
    raise

logger = logging.getLogger(__name__)

class RecognitionWorker(threading.Thread):
    def __init__(self, task_queue, result_queue, stop_event, pipeline):
        super().__init__(daemon=True)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        # 接收主进程注入的 C++ Pipeline
        self.pipeline = pipeline 

    def run(self):
        logger.info(f"[OCR Thread-{self.name}] C++ Native NPU 引擎就绪，等待任务...")
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None: 
                    break
                    
                track_id, crop_img = task
                
                # 调用 C++ 底层，GIL 会在此处自动释放，实现真正的硬件并发
                res = self.pipeline.process(crop_img)
                
                # 如果 C++ 返回了有效的识别结果
                if res.color_type != "unknown" and len(res.landmarks) == 8:
                    # 将 C++ 传回的 1D list 重新 reshape 为 Engine 需要的 (4, 2) 归一化坐标矩阵
                    rel_landmarks = np.array(res.landmarks, dtype=np.float32).reshape(4, 2)
                    self.result_queue.put((track_id, res.color_type, res.confidence, rel_landmarks))
                    
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[OCR Thread-{self.name}] 运行异常: {e}")


class AsyncPlateRecognizer:
    def __init__(self, config, num_workers=2):
        self.cfg = config
        self.task_queue = queue.Queue(maxsize=3) 
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.workers = []
        
        logger.info("[Async OCR] 正在初始化 C++ Hailo NPU Pipeline...")
        # 不再实例化 Python 的 Pipeline，而是直接实例化 C++ 硬件接管类
        self.pipeline = hailo_ocr_cpp.HailoPlateRecognizer(self.cfg.Y5FU_PATH, self.cfg.LITEMODEL_PATH)
        
        for _ in range(num_workers):
            # 将 C++ pipeline 当作共享资源塞给所有的 Worker 线程
            w = RecognitionWorker(self.task_queue, self.result_queue, self.stop_event, self.pipeline)
            w.start()
            self.workers.append(w)

    def push_task(self, track_id: int, image: np.ndarray) -> bool:
        try:
            self.task_queue.put_nowait((track_id, image))
            return True
        except queue.Full:
            return False

    def get_results(self) -> list:
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def stop(self):
        logger.info("正在停止所有 OCR 线程...")
        self.stop_event.set()
        
        for _ in self.workers:
            try:
                self.task_queue.put_nowait(None)
            except queue.Full:
                pass
                
        for w in self.workers:
            w.join(timeout=2.0)
            
        # 注意：在 C++ 版本中，我们不需要手动调 self.pipeline.release()
        # 因为当 AsyncPlateRecognizer 被销毁时，self.pipeline 会被 Python 的垃圾回收处理
        # 从而触发 C++ 端的 ~HailoPlateRecognizer() 析构函数，安全、干净地释放 NPU 资源。
