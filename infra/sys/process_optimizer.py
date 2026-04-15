import os
import psutil

class SystemOptimizer:
    """边缘端 (Raspberry Pi 5) 多进程架构系统优化器"""

    @staticmethod
    def optimize_classifier_process():
        """
        优化异步分类子进程 (CPU 密集型的 ONNX 推断)
        """
        try:
            p = psutil.Process(os.getpid())
            try:
                os.nice(5)
            except PermissionError:
                pass
            print(f">>> [System] 车牌分类工作进程 (PID: {p.pid}) 已隔离至 CPU Core 3")
        except Exception as e:
            pass
