import os
import psutil

class SystemOptimizer:
    """边缘端 (Raspberry Pi 5) 多进程架构系统优化器"""

    @staticmethod
    def optimize_main_process():
        """
        优化主进程 (负责视频流、追踪、UI渲染、物理计算)
        策略: 绑定至 Core 2，并提升进程优先级以保证画面绝对流畅。
        """
        try:
            p = psutil.Process(os.getpid())
            # 绑定到 Core 2 (留出 Core 0, 1 给 OS 和 GStreamer)
            p.cpu_affinity([2])
            
            # 提升优先级 (Linux nice value: -10，越小优先级越高，需要 sudo)
            try:
                os.nice(-10)
            except PermissionError:
                print(">>> [System Warn] 提升主进程优先级需要 sudo 权限，建议使用 sudo 运行程序。")
                
            print(f">>> [System] 🚀 主进程 (PID: {p.pid}) 已绑定至 CPU Core 2 | 优先级已提升")
        except Exception as e:
            print(f">>> [System Warn] 主进程优化失败: {e}")

    @staticmethod
    def optimize_classifier_process():
        """
        优化异步分类子进程 (CPU 密集型的 ONNX 推断)
        """
        try:
            p = psutil.Process(os.getpid())
            p.cpu_affinity([3]) # 绑定至 Core 3
            try:
                os.nice(5)
            except PermissionError:
                pass
            print(f">>> [System] 🚜 车牌分类工作进程 (PID: {p.pid}) 已隔离至 CPU Core 3")
        except Exception as e:
            pass
