import numpy as np
from multiprocessing import shared_memory
import queue
import logging

logger = logging.getLogger(__name__)

class SharedMemoryPool:
    """环形共享内存池，用于主进程与 Worker 进程间的图像零拷贝传输"""
    
    def __init__(self, pool_size=10, max_width=256, max_height=256, channels=3, dtype=np.uint8):
        self.pool_size = pool_size
        # 使用真实的 dtype.itemsize 来计算物理字节上限，杜绝越界
        self.block_bytes = max_width * max_height * channels * np.dtype(dtype).itemsize
        
        self.shm_blocks = []
        # 注意：这是标准库队列，仅限主进程管理和使用，跨进程请用 mp.Queue
        self.free_indices = queue.Queue()

        # 初始化 N 块固定大小的共享内存
        for i in range(self.pool_size):
            shm_name = f"ocr_shm_block_{i}"
            try:
                # 尝试清理上次异常退出残留的同名内存块
                existing_shm = shared_memory.SharedMemory(name=shm_name)
                existing_shm.unlink()
            except FileNotFoundError:
                pass
                
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.block_bytes)
            self.shm_blocks.append(shm)
            self.free_indices.put(i) # 初始时所有块均空闲
            
        logger.info(f"成功初始化零拷贝共享内存池: {pool_size} blocks, {self.block_bytes/1024:.1f} KB/block")

    def allocate_and_write(self, image: np.ndarray):
        """
        获取一个空闲块并写入图像。
        返回 (idx, shape, dtype_str)，如果池满则返回 None。
        """
        # 1. 安全校验：基于物理字节数，兼容单通道灰度图或浮点图
        if image.nbytes > self.block_bytes:
            logger.error(f"图像尺寸超出共享内存上限! 需求: {image.nbytes}B, 上限: {self.block_bytes}B")
            return None

        try:
            # 2. 获取空闲索引（非阻塞，拿不到说明池满了，限流丢弃）
            idx = self.free_indices.get_nowait()
        except queue.Empty:
            # 缓冲池打满是边缘端的常态，直接静默丢弃即可，无需报错
            return None 

        # 3. 内存拷贝
        shm = self.shm_blocks[idx]
        # 根据实际图像的 shape 和 dtype 创建一个指向共享内存的 Numpy 视图
        shm_array = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
        np.copyto(shm_array, image) 
        
        return idx, image.shape, image.dtype.str

    def free_block(self, idx: int):
        """
        归还内存块使用权。
        子进程通过 IPC Queue 将 idx 发回给主进程，由主进程调用此方法。
        """
        self.free_indices.put(idx)

    def cleanup(self):
        """系统退出时销毁所有共享内存"""
        for i, shm in enumerate(self.shm_blocks):
            try:
                shm.close()
                shm.unlink()
            except Exception as e:
                # 忽略清理时的重复释放错误
                pass
        logger.info("共享内存池已安全释放。")
