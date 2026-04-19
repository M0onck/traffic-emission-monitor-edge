import os
import psutil
import logging

logger = logging.getLogger(__name__)

class ProcessOptimizer:
    """边缘端 (Raspberry Pi 5) 多核物理隔离与系统优化器"""

    @staticmethod
    def optimize_main_process():
        """
        [主进程护城河]
        将主程序及其未来派生的所有子线程（如 GStreamer, 热成像看门狗, NPU 底层）
        死死封印在 Core 0, 1, 2 上，绝对禁止它们进入 Core 3。
        """
        try:
            p = psutil.Process(os.getpid())
            # 核心防御：绑定到物理核心 0, 1, 2
            p.cpu_affinity([0, 1, 2])
            print(f">>> [System] 主引擎 (PID: {p.pid}) 已成功锁定至 CPU Core 0, 1, 2")
        except Exception as e:
            print(f">>> [System-Error] 主引擎 CPU 亲和度设置失败: {e}")

    @staticmethod
    def optimize_classifier_process():
        """
        [OCR 黑盒囚笼]
        将极其消耗算力的 ONNX 推断进程死死锁在 Core 3 上，
        它无论怎么满载，都绝对跨越不了物理边界。
        """
        try:
            p = psutil.Process(os.getpid())
            # 核心防御：绑定到物理核心 3
            p.cpu_affinity([3])
            
            # 降低优先级（Nice 值为正，优先级降低），确保即使内核在 Core 3 有硬中断，也能让内核优先
            try:
                os.nice(5)
            except PermissionError:
                pass
                
            print(f">>> [System] OCR 模块 (PID: {p.pid}) 已隔离至 CPU Core 3")
        except Exception as e:
            print(f">>> [System-Error] OCR 工作进程 CPU 亲和度设置失败: {e}")
