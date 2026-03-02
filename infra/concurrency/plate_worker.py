import threading  # 🌟 核心变化：使用线程替代进程
import queue      # 🌟 核心变化：使用标准队列替代跨进程队列
import numpy as np
import os

from perception.plate_classifier.pipeline import EdgePlateClassifierPipeline
from perception.plate_classifier.core.hailo_support import MultiTaskDetectorHailo, ClassificationHailo
from infra.sys.process_optimizer import SystemOptimizer

def _worker_process(task_queue, result_queue, detector_hef_path, classifier_hef_path):
    """
    运行在独立线程上的工作函数 (与 GStreamer 共享同一个进程 PID)
    """
    # 线程内绑核可能会影响整个进程，安全起见我们先将其注释掉
    # SystemOptimizer.optimize_classifier_process()
    
    # 绝对确保没有这个环境变量干扰
    os.environ.pop("HAILORT_USE_SERVICE", None)

    print(">>> [PlateWorker-Thread] 正在线程中连接 Hailo NPU...")
    try:
        from hailo_platform import VDevice
        # 配置同组 ID，触发 HailoRT 进程内原生复用机制
        params = VDevice.create_params()
        params.group_id = "1"
        target_vdevice = VDevice(params) 
        
        detector = MultiTaskDetectorHailo(hef_path=detector_hef_path, target_vdevice=target_vdevice)
        classifier = ClassificationHailo(hef_path=classifier_hef_path, target_vdevice=target_vdevice)
        
        pipeline = EdgePlateClassifierPipeline(detector, classifier)
        print(">>> [PlateWorker-Thread] NPU 算力已就绪，多模型并发开启。")
        
    except Exception as e:
        print(f">>> [PlateWorker 致命错误] NPU 初始化失败: {e}")
        return
    
    # 消费循环
    while True:
        task = task_queue.get()
        if task is None: 
            break
            
        tid, crop_img = task
        if crop_img is None or crop_img.size == 0:
            continue
            
        try:
            # 硬件加速推理
            color_type, conf, _ = pipeline.process(crop_img)
            result_queue.put((tid, color_type, conf))
        except Exception as e:
            print(f"[PlateWorker 推理错误] {e}")


class PlateClassifierWorker:
    """车牌属性分类的异步工作管理器 (基于多线程)"""
    
    def __init__(self, detector_hef_path, classifier_hef_path, max_queue_size=10):
        # 🌟 换成轻量级的线程安全队列
        self.task_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        
        # 🌟 创建守护线程
        self.thread = threading.Thread(
            target=_worker_process,
            args=(self.task_queue, self.result_queue, detector_hef_path, classifier_hef_path),
            daemon=True
        )
    
    def start(self):
        self.thread.start()
        
    def stop(self):
        try:
            self.task_queue.put_nowait(None) 
        except queue.Full:
            pass
        self.thread.join(timeout=2)
        
    def push_task(self, tid, crop_img: np.ndarray) -> bool:
        try:
            self.task_queue.put_nowait((tid, crop_img))
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
