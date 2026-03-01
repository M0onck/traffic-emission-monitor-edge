import os
import psutil

class SystemOptimizer:
    """边缘端 (Raspberry Pi 5) 进程优化器"""
    @staticmethod
    def set_cpu_affinity(role="main"):
        try:
            p = psutil.Process(os.getpid())
            # 树莓派5拥有 0, 1, 2, 3 四个核心
            if role == "main":
                # 将 Python 主循环(物理模型计算、数据库写入等)绑定到 Core 3
                # 留出 Core 0-2 给 GStreamer 及底层驱动调度
                p.cpu_affinity([3])
                print(">>> [System] 主线程已绑定至 CPU Core 3")
        except Exception as e:
            print(f">>> [System Warn] CPU 亲和性设置失败 (忽略该警告如果你不在 Linux 环境下运行): {e}")
