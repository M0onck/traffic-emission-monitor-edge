import numpy as np
from multiprocessing import shared_memory, Lock
import queue

class SharedMemoryPool:
    """环形共享内存池，用于多进程间的图像零拷贝传输"""
    def __init__(self, pool_size=10, max_width=256, max_height=256, channels=3):
        self.pool_size = pool_size
        self.block_bytes = max_width * max_height * channels
        self.shape_limit = (max_height, max_width, channels)
        
        self.shm_blocks = []
        self.free_indices = queue.Queue()
        self.lock = Lock()

        # 初始化 N 块固定大小的共享内存
        for i in range(self.pool_size):
            shm_name = f"ocr_shm_block_{i}"
            try:
                # 尝试清理残留的同名内存块
                existing_shm = shared_memory.SharedMemory(name=shm_name)
                existing_shm.unlink()
            except FileNotFoundError:
                pass
                
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.block_bytes)
            self.shm_blocks.append(shm)
            self.free_indices.put(i) # 初始时所有块均空闲

    def allocate_and_write(self, image: np.ndarray):
        """主进程调用：获取一个空闲块并写入图像"""
        h, w, c = image.shape
        if h > self.shape_limit[0] or w > self.shape_limit[1]:
            raise ValueError("图像尺寸超出共享内存块预设上限")

        try:
            # 获取空闲索引（非阻塞，拿不到说明池满了，直接丢弃该帧防堆积）
            idx = self.free_indices.get_nowait()
        except queue.Empty:
            return None # 缓冲池满，丢弃任务

        # 写入数据
        shm = self.shm_blocks[idx]
        shm_array = np.ndarray((h, w, c), dtype=image.dtype, buffer=shm.buf)
        np.copyto(shm_array, image) # 内存拷贝
        
        return idx, (h, w, c), image.dtype.str

    def free_block(self, idx: int):
        """主进程或子进程调用：释放内存块回内存池"""
        self.free_indices.put(idx)

    def cleanup(self):
        """系统退出时销毁所有共享内存"""
        for shm in self.shm_blocks:
            shm.close()
            shm.unlink()
