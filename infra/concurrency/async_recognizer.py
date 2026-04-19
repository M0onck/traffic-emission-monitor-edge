import threading
import queue
import logging
import numpy as np

# 引入原生的 Python 版 ONNX 管道及类型定义
from perception.plate_classifier.pipeline import EdgePlateClassifierPipeline
from perception.plate_classifier.core.multitask_detect import MultiTaskDetectorORT
from perception.plate_classifier.core.classification import ClassificationORT
from perception.plate_classifier.core.typedef import UNKNOWN, GREEN, BLUE, YELLOW_SINGLE

logger = logging.getLogger(__name__)

class RecognitionWorker(threading.Thread):
    def __init__(self, task_queue, result_queue, stop_event, cfg):
        super().__init__(daemon=True)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.cfg = cfg
        self.pipeline = None

    def run(self):
        logger.info(f"[OCR Thread-{self.name}] 正在初始化 ONNX 模型 (独立实例以保证线程安全)...")
        # 为每个线程单独实例化模型，完美解决 tmp_pack 导致的竞态条件
        detector = MultiTaskDetectorORT(self.cfg.Y5FU_PATH)
        classifier = ClassificationORT(self.cfg.LITEMODEL_PATH)
        self.pipeline = EdgePlateClassifierPipeline(detector, classifier)
        
        logger.info(f"[OCR Thread-{self.name}] ONNX 引擎就绪，等待任务...")
        
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None: 
                    break
                    
                track_id, crop_img = task
                crop_h, crop_w = crop_img.shape[:2]
                
                # 调用纯 Python ONNX 管道 (此处 ONNXRuntime 底层会自动释放 GIL)
                plate_type, conf, plate_box, plate_points = self.pipeline.process(crop_img)
                
                # 只有识别成功才推入结果队列
                if plate_type != UNKNOWN and plate_points is not None:
                    # 1. 映射整数分类标志为引擎所需的字符串格式
                    color_type = "unknown"
                    if plate_type == GREEN: color_type = "green"
                    elif plate_type == BLUE: color_type = "blue"
                    elif plate_type == YELLOW_SINGLE: color_type = "yellow"
                    
                    # 2. 将绝对像素坐标系转换为 Engine 期待的 (0~1) 相对坐标
                    rel_landmarks = plate_points.astype(np.float32)
                    rel_landmarks[:, 0] /= float(crop_w)
                    rel_landmarks[:, 1] /= float(crop_h)
                    
                    self.result_queue.put((track_id, color_type, float(conf), rel_landmarks))
                    
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
        
        logger.info("[Async OCR] 启动基于 ONNXRuntime 的异步车牌识别池...")
        
        for _ in range(num_workers):
            # 将 cfg 直接传入 Worker，由 Worker 自己内部创建 Pipeline
            w = RecognitionWorker(self.task_queue, self.result_queue, self.stop_event, self.cfg)
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
