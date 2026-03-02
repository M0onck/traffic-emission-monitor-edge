import multiprocessing as mp
import queue
import numpy as np

# 延迟导入模型，确保它们在子进程中被加载
from perception.plate_classifier.pipeline import EdgePlateClassifierPipeline
from perception.plate_classifier.core.multitask_detect import MultiTaskDetectorORT
from perception.plate_classifier.core.classification import ClassificationORT
from infra.sys.process_optimizer import SystemOptimizer

def _worker_process(task_queue, result_queue, detector_path, classifier_path):
    """
    运行在独立 CPU 核心 (Core 3) 上的子进程主函数
    """
    # 1. 进程优化：绑定到指定的核心
    SystemOptimizer.optimize_classifier_process()
    
    # 2. 核心：在子进程内部初始化 ONNX 模型，避免序列化报错
    print(">>> [PlateWorker] 正在子进程中初始化分类模型...")
    detector = MultiTaskDetectorORT(detector_path)
    classifier = ClassificationORT(classifier_path)
    pipeline = EdgePlateClassifierPipeline(detector, classifier)
    print(">>> [PlateWorker] 模型初始化完成，开始监听任务。")
    
    # 3. 消费循环
    while True:
        task = task_queue.get()
        if task is None:  # 毒丸(Poison Pill) 退出机制
            break
            
        tid, crop_img = task
        if crop_img is None or crop_img.size == 0:
            continue
            
        # 进行推理计算 (CPU 密集型操作)
        color_type, conf, _ = pipeline.process(crop_img)
        
        # 仅回传结果 (IPC 传输极小量的数据)
        result_queue.put((tid, color_type, conf))


class PlateClassifierWorker:
    """车牌属性分类的异步工作管理器 (运行于主进程)"""
    
    def __init__(self, detector_path, classifier_path, max_queue_size=10):
        # 使用 max_queue_size 防止处理不过来导致内存爆炸
        self.task_queue = mp.Queue(maxsize=max_queue_size)
        self.result_queue = mp.Queue()
        
        self.process = mp.Process(
            target=_worker_process,
            args=(self.task_queue, self.result_queue, detector_path, classifier_path),
            daemon=True
        )
    
    def start(self):
        self.process.start()
        
    def stop(self):
        try:
            self.task_queue.put_nowait(None) # 发送毒丸
        except queue.Full:
            pass
        self.process.join(timeout=2)
        
    def push_task(self, tid, crop_img: np.ndarray) -> bool:
        """非阻塞地将图片送入队列"""
        try:
            self.task_queue.put_nowait((tid, crop_img))
            return True
        except queue.Full:
            return False # 队列满了就直接丢弃，不阻塞主线程
            
    def get_results(self) -> list:
        """非阻塞地收割所有已完成的结果"""
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results
