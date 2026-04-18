import multiprocessing as mp
import ctypes
import numpy as np
import threading
import time
import os
import logging

logger = logging.getLogger(__name__)

def _thermal_worker(lib_path, shared_array, heartbeat, run_flag):
    """
    隔离子进程
    即使在这里发生 I2C 底层死锁，也不会波及主系统。
    """
    try:
        lib = ctypes.cdll.LoadLibrary(lib_path)
        lib.get_mlx90640_temp.argtypes = [ctypes.POINTER(ctypes.c_float)]
    except Exception as e:
        logger.error(f"[ThermalWorker] 库加载失败: {e}")
        return

    # 剥离包装器，提取原生 C 数组，并转换为指针
    raw_array = shared_array.get_obj()
    ptemp = ctypes.cast(raw_array, ctypes.POINTER(ctypes.c_float))
    
    # 显式定义 C 函数的返回值类型为整型
    lib.get_mlx90640_temp.restype = ctypes.c_int

    while run_flag.value:
        try:
            # C++ 包装层在成功时返回 0，失败时返回负数
            status = lib.get_mlx90640_temp(ptemp)
            
            if status == 0:
                # 只有真正拿到数据，才喂狗
                heartbeat.value = time.time()
            else:
                logger.error(f"[ThermalWorker] 底层 I2C 读取失败，返回码: {status}")
                time.sleep(0.1) # 防止死循环耗尽 CPU

        except Exception as e:
            logger.error(f"[ThermalWorker] 致命调用异常: {e}")
            time.sleep(0.5)

class ThermalCamera:
    """
    具备自动恢复能力的热成像驱动封装
    """
    def __init__(self, lib_path="bin/libmlx90640.so"):
        # 将传入的路径转换为绝对路径，并绑定到 self.lib_path
        self.lib_path = os.path.abspath(lib_path)
        
        # 使用 self.lib_path 进行判断和加载
        if not os.path.exists(self.lib_path):
            print(f">>> [Warning] 热成像动态库未找到: {self.lib_path}")
            self.lib = None
            return
            
        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        
        # 1. 开辟进程间共享内存
        self.shared_array = mp.Array(ctypes.c_float, 768)
        self.heartbeat = mp.Value('d', time.time()) # 双精度浮点型时间戳
        self.run_flag = mp.Value(ctypes.c_bool, False)
        
        self._process = None
        self._watchdog_thread = None
        
        # 记录重启次数，防止陷入无限死循环
        self._restart_count = 0
        self._max_restarts = 5 

    def start(self):
        if not os.path.exists(self.lib_path):
            logger.warning(f"热成像动态库未找到: {self.lib_path}，模块已停用.")
            return
            
        self.run_flag.value = True
        self._start_worker_process()
        
        # 启动后台看门狗线程监控子进程
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _start_worker_process(self):
        """拉起硬件工作进程"""
        self.heartbeat.value = time.time() # 启动前刷新心跳
        self._process = mp.Process(
            target=_thermal_worker, 
            args=(self.lib_path, self.shared_array, self.heartbeat, self.run_flag),
            daemon=True
        )
        self._process.start()
        logger.info(f"[ThermalCamera] 热成像工作进程已启动 (PID: {self._process.pid})")

    def _watchdog_loop(self):
        while self.run_flag.value:
            time.sleep(3.0) 
            
            time_since_last_beat = time.time() - self.heartbeat.value
            
            # 超过 3 秒没有心跳（说明底层连续 3 秒都在报错返回 -1）
            if time_since_last_beat > 3.0:
                if self._restart_count >= self._max_restarts:
                    logger.critical("[ThermalCamera] 连续重启达到上限，热成像硬件可能已物理断开.")
                    self.run_flag.value = False
                    break
                    
                logger.warning(f"[ThermalCamera] 检测到 I2C 总线挂起或持续报错 (丢失心跳 {time_since_last_beat:.1f}s)。正在尝试硬重置...")
                self._restart_count += 1
                
                # 强杀挂起的子进程释放 /dev/i2c-1 的文件句柄
                if self._process and self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=1.0)
                    if self._process.is_alive():
                        os.kill(self._process.pid, 9) # 终极猎杀
                
                # 给 Linux 内核一点时间清理 I2C 驱动状态
                time.sleep(1.5) 
                
                self._start_worker_process()
                logger.info(f"[ThermalCamera] 自动重启完成 (当前重试: {self._restart_count}/{self._max_restarts})")
            
            # 如果成功读取，慢慢重置重启计数器
            elif time_since_last_beat < 1.0 and self._restart_count > 0:
                self._restart_count = 0

    def read(self) -> np.ndarray:
        """主引擎调用的非阻塞读取接口"""
        if not self.run_flag.value or self._process is None or not self._process.is_alive():
            return None
            
        # 如果心跳严重超时，返回 None 避免主引擎拿到陈旧的冻结画面
        if time.time() - self.heartbeat.value > 2.0:
            return None
            
        # 极速零拷贝：直接从共享内存映射出 24x32 矩阵
        return np.frombuffer(self.shared_array.get_obj(), dtype=np.float32).reshape((24, 32))

    def stop(self):
        self.run_flag.value = False
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        logger.info("[ThermalCamera] 热成像模块已安全停止.")
