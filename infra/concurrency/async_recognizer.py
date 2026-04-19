import multiprocessing as mp
import queue
import logging
import numpy as np
import cv2
import os

# 引入原生的 Python 版 ONNX 管道及类型定义
from perception.plate_classifier.pipeline import EdgePlateClassifierPipeline
from perception.plate_classifier.core.multitask_detect import MultiTaskDetectorORT
from perception.plate_classifier.core.classification import ClassificationORT
from perception.plate_classifier.core.typedef import UNKNOWN, GREEN, BLUE, YELLOW_SINGLE

logger = logging.getLogger(__name__)

# ==========================================
# 核心修改 1：继承 mp.Process 而不是 threading.Thread
# ==========================================
class RecognitionWorker(mp.Process):
    def __init__(self, task_queue, result_queue, stop_event, cfg, worker_id):
        super().__init__(daemon=True)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.cfg = cfg
        self.worker_id = worker_id

    def run(self):
        # ========================================================
        # 终极物理隔离：强制将 OCR 进程锁死在树莓派的 Core 3 上
        # 它永远无法跨越物理边界去抢占主引擎(Core 2)或底层驱动(Core 0/1)的算力
        # ========================================================
        if hasattr(os, 'sched_setaffinity'):
            core_id = 3
            os.sched_setaffinity(0, {core_id})
            logger.info(f"[OCR Worker-{self.worker_id}] 物理隔离生效，已成功锁定至 CPU 核心 {core_id}")
            
        # 限制底层的隐式多线程抢占
        cv2.setNumThreads(1)
        os.environ["OMP_NUM_THREADS"] = "1"

        logger.info(f"[OCR Worker-{self.worker_id}] 正在初始化 ONNX 模型...")
        # 为每个进程单独实例化模型，完美避开多进程/多线程竞态条件
        detector = MultiTaskDetectorORT(self.cfg.Y5FU_PATH)
        classifier = ClassificationORT(self.cfg.LITEMODEL_PATH)
        pipeline = EdgePlateClassifierPipeline(detector, classifier)
        
        logger.info(f"[OCR Worker-{self.worker_id}] ONNX 引擎就绪，等待任务...")
        
        BLUR_THRESHOLD = getattr(self.cfg, 'BLUR_THRESHOLD', 100.0)
        
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None: 
                    break
                    
                track_id, vehicle_img = task
                crop_h, crop_w = vehicle_img.shape[:2]
                
                # ==========================================================
                # 核心修改 2：从主引擎接管过来的算力包袱（图像质量评估与预处理）
                # ==========================================================
                # 1. 拉普拉斯方差清晰度评估
                gray_crop = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
                
                if blur_score < BLUR_THRESHOLD:
                    continue # 画面太糊，静默丢弃，避免浪费 ONNX 算力
                    
                # 2. CLAHE 增强预处理
                lab = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl,a,b))
                enhanced_crop = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

                # ==========================================================
                # 调用纯 Python ONNX 管道进行推理
                # ==========================================================
                result = pipeline.process(enhanced_crop)
                
                # 兼容未检测到车牌时返回 3 个值的情况
                if len(result) == 3:
                    plate_type, conf, plate_points = result
                else:
                    plate_type, conf, plate_box, plate_points = result
                
                # 只有识别成功才推入结果队列
                if plate_type != UNKNOWN and plate_points is not None:
                    color_type = "unknown"
                    if plate_type == GREEN: color_type = "green"
                    elif plate_type == BLUE: color_type = "blue"
                    elif plate_type == YELLOW_SINGLE: color_type = "yellow"
                    
                    # 将绝对像素坐标系转换为相对坐标
                    rel_landmarks = plate_points.astype(np.float32)
                    rel_landmarks[:, 0] /= float(crop_w)
                    rel_landmarks[:, 1] /= float(crop_h)
                    
                    self.result_queue.put((track_id, color_type, float(conf), rel_landmarks))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[OCR Worker-{self.worker_id}] 运行异常: {e}")


class AsyncPlateRecognizer:
    def __init__(self, config, num_workers=1):
        self.cfg = config
        
        # ==========================================
        # 核心修改 3：使用 spawn 上下文创建跨进程安全的队列和事件
        # ==========================================
        ctx = mp.get_context('spawn')
        self.task_queue = ctx.Queue(maxsize=10) # 稍微加大缓冲池，防止主线程塞满阻塞
        self.result_queue = ctx.Queue()
        self.stop_event = ctx.Event()
        self.workers = []
        
        logger.info(f"[Async OCR] 启动 {num_workers} 个独立 Spawn 进程的异步车牌识别池...")
        
        for i in range(num_workers):
            w = RecognitionWorker(self.task_queue, self.result_queue, self.stop_event, self.cfg, worker_id=i)
            w.start()
            self.workers.append(w)

    def push_task(self, track_id: int, image: np.ndarray) -> bool:
        try:
            # np.ascontiguousarray 确保图像内存连续，防止跨进程 Pickle 序列化时崩溃
            self.task_queue.put_nowait((track_id, np.ascontiguousarray(image)))
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
        logger.info("正在停止所有 OCR 进程...")
        self.stop_event.set()
        
        # 向队列推入 None 作为退出毒丸 (Poison Pill)
        for _ in self.workers:
            try:
                self.task_queue.put_nowait(None)
            except queue.Full:
                pass
                
        for w in self.workers:
            w.join(timeout=2.0)
            if w.is_alive():
                w.terminate() # 如果 2 秒后进程没退出，强制击杀
