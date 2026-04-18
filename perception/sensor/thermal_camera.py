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
    except Exception as e:
        logger.error(f"[ThermalWorker] 库加载失败: {e}")
        return

    ptemp = ctypes.pointer(shared_array)
    
    while run_flag.value:
        try:
            # 阻塞型硬件读取
            lib.get_mlx90640_temp(ptemp)
            # 读取成功，立刻更新心跳时间戳
            heartbeat.value = time.time()
        except Exception as e:
            # 过滤偶发的传输错误，防止频发刷屏
            time.sleep(0.5)

class ThermalCamera:
    """
    具备自动恢复能力的热成像驱动封装
    """
    def __init__(self, lib_path='./libmlx90640.so'):
        self.lib_path = lib_path
        
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
        """
        看门狗守护逻辑
        监控硬件进程状态，实现自动断线重连。
        """
        while self.run_flag.value:
            time.sleep(2.0) # 每2秒巡视一次
            
            # 计算距离上次成功读取的时间差
            time_since_last_beat = time.time() - self.heartbeat.value
            
            # 如果超过 5 秒没有心跳，判定为 I2C 死锁
            if time_since_last_beat > 5.0:
                if self._restart_count >= self._max_restarts:
                    logger.error("[ThermalCamera] I2C 硬件连续重启失败，已触发断路，停止热成像采集.")
                    self.run_flag.value = False # 放弃该传感器，让系统主线继续运行
                    break
                    
                logger.warning(f"[ThermalCamera] 检测到 I2C 死锁 (心跳停止 {time_since_last_beat:.1f}s)。正在尝试自动重启...")
                self._restart_count += 1
                
                # 1. 强制猎杀卡死的进程，释放 /dev/i2c 文件描述符
                if self._process and self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=1.0)
                
                # 2. 稍微给 Linux 内核一点时间清理底层驱动状态
                time.sleep(1.0) 
                
                # 3. 重新拉起进程
                self._start_worker_process()
                logger.info(f"[ThermalCamera] 自动重启完成，已重置 I2C 连接 (当前重试: {self._restart_count}/{self._max_restarts})")
            
            # 如果成功读取，慢慢重置重启计数器（证明稳定下来了）
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
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
        logger.info("[ThermalCamera] 热成像模块已安全停止.")
