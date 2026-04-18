import cv2
import numpy as np
import multiprocessing as mp
import queue
import logging
from perception.plate_classifier.core.multitask_detect import letter_box, post_precessing
from perception.plate_classifier.core.hailo_support import MultiTaskDetectorHailo, ClassificationHailo
from infra.concurrency.shared_memory_pool import SharedMemoryPool

logger = logging.getLogger(__name__)

class AsyncPlateRecognizer:
    def __init__(self, y5fu_onnx_path, litemodel_onnx_path, num_workers=1):
        ctx = mp.get_context('spawn')
        self.task_queue = ctx.Queue(maxsize=30)
        self.result_queue = ctx.Queue()
        self.workers = [] # 保存子进程的引用

        # 实例化共享内存池，容量与 task_queue 保持一致
        # 上限应该适应大卡车或近景车辆的截图
        self.shm_pool = SharedMemoryPool(pool_size=30, max_width=1280, max_height=1280, channels=3)

        print(f">>> [AsyncRecognizer] 启动 {num_workers} 个独立 Spawn 进程 ONNX 引擎...")

        # 启动多个并行的 Worker 进程
        for i in range(num_workers):
            worker_process = ctx.Process(
                target=self._worker_loop,
                args=(self.task_queue, self.result_queue, y5fu_onnx_path, litemodel_onnx_path, i),
                daemon=True
            )
            worker_process.start()
            self.workers.append(worker_process)

    def push_task(self, track_id, vehicle_crop):
        # 1. 尝试写入共享内存
        alloc_result = self.shm_pool.allocate_and_write(vehicle_crop)
        if alloc_result is None:
            return False # 内存池满，限流丢弃

        idx, shape, dtype_str = alloc_result

        # 2. 队列中只传递轻量的元数据
        try:
            self.task_queue.put_nowait((track_id, idx, shape, dtype_str))
            return True
        except queue.Full:
            self.shm_pool.free_block(idx) # 如果队列满，归还内存块
            return False

    def get_results(self):
        results = []
        while not self.result_queue.empty():
            try:
                res = self.result_queue.get_nowait()
                
                # 拦截子进程发来的释放信号，完成内存池闭环
                if isinstance(res, tuple) and res[0] == "FREE_BLOCK":
                    shm_idx = res[1]
                    self.shm_pool.free_block(shm_idx)
                else:
                    # 正常的识别结果放入列表返回给主循环
                    results.append(res)
                    
            except queue.Empty:
                break
        return results

    def stop(self):
        """安全终止所有的 OCR 工作进程"""
        print("[AsyncRecognizer] 正在发送安全退出信号...")
        for _ in self.workers:
            try:
                self.task_queue.put_nowait(("POISON_PILL", None))
            except queue.Full:
                pass 

        for worker in self.workers:
            worker.join(timeout=3.0)
            if worker.is_alive():
                worker.terminate() # 仅作最后兜底
                worker.join(timeout=1.0)

        # 清理系统底层挂载的共享内存块，防泄漏
        if hasattr(self, 'shm_pool'):
            self.shm_pool.cleanup()
        
        print("[AsyncRecognizer] 所有 Hailo OCR 子进程已安全销毁。")

    @staticmethod
    def _worker_loop(task_queue, result_queue, y5fu_path, lite_path, worker_id):
        import os
        import sys
        import time
        import queue
        from multiprocessing import shared_memory
        import logging

        # 为 Spawn 的子进程配置基础日志格式
        logging.basicConfig(
            level=logging.DEBUG, 
            format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        logger = logging.getLogger(f"Worker-{worker_id}")

        # CPU 绑核与高优先级分配
        try:
            # 树莓派 5 有 4 个核心 (0, 1, 2, 3)
            # 将 worker_id 映射到对应的核心上，防止被其他繁重线程挤占
            core_id = worker_id % os.cpu_count()
            os.sched_setaffinity(0, {core_id})
            
            # 尝试提高当前 NPU 进程的 Linux 调度优先级 (-5 比默认的 0 优先级高)
            os.nice(-5)
        except AttributeError:
            pass # 兼容非 Linux 系统
        except PermissionError:
            pass # 如果没有 sudo 权限，nice 会报错，忽略即可

        # 1. 引入 Hailo 虚拟设备管理器
        from hailo_platform import VDevice, InferVStreams, HailoSchedulingAlgorithm

        # 2. 创建允许跨进程共享的参数
        params = VDevice.create_params()
        params.group_id = "SHARED" 
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.multi_process_service = True 
        
        # 3. 在子进程内部接管 NPU 硬件资源
        with VDevice(params) as target_vdevice:
            logger.info("已成功挂载 Hailo VDevice 并开始加载 HEF...")
            
            # 4. 实例化 NPU 推理引擎
            detector = MultiTaskDetectorHailo(y5fu_path, target_vdevice)
            classifier = ClassificationHailo(lite_path, target_vdevice)

            det_args = detector.get_pipeline_args()
            cls_args = classifier.get_pipeline_args()

            colors = ["blue", "green", "yellow", "white", "black"]
            dst_pts = np.array([[0, 0], [96, 0], [96, 32], [0, 32]], dtype=np.float32)

            # 外层循环：用于在 NPU 异常时自动重建 VStream 管道
            while True:
                try:
                    with InferVStreams(*det_args) as det_pipe, InferVStreams(*cls_args) as cls_pipe:
                        logger.info("所有的 NPU 推理通道已成功安全建立！")

                        while True:
                            # ================= 阶段 A：任务获取与解析 =================
                            try:
                                # 1. 整体接收数据，防死锁超时
                                task_data = task_queue.get(timeout=1.0)
                            except queue.Empty:
                                continue
                            except Exception:
                                continue

                            # 2. 优先检查毒药丸，防解包崩溃 (主进程发来的通常是 ("POISON_PILL", None))
                            if task_data[0] == "POISON_PILL":
                                logger.info("收到安全停机指令，正在释放 NPU 并销毁进程...")
                                return # 触发 context manager 的 __exit__ 清理硬件

                            # 3. 解析正常任务的元数据
                            try:
                                track_id, shm_idx, shape, dtype_str = task_data
                            except ValueError:
                                logger.error(f"任务载荷异常，预期 4 个参数，实际收到: {task_data}")
                                continue

                            # ================= 阶段 B：零拷贝读取内存池 =================
                            vehicle_img = None
                            shm = None
                            try:
                                # 挂载主进程分配好的共享内存块
                                shm_name = f"ocr_shm_block_{shm_idx}"
                                shm = shared_memory.SharedMemory(name=shm_name)

                                # 使用 np.ndarray 映射内存，并立即调用 .copy() 
                                # copy() 会将数据搬运到当前进程的独立内存中，从而切断对共享内存的依赖
                                vehicle_img = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf).copy()

                            except Exception as e:
                                logger.error(f"读取共享内存失败 TID={track_id}: {e}")
                                continue
                            finally:
                                # 【极度关键的清理工作】
                                # 1. 立即关闭当前进程对共享内存的挂载，防止文件句柄（FD）泄漏
                                if shm is not None:
                                    shm.close() 
                                
                                # 2. 通过结果队列通知主进程：该索引的内存块已提取完毕，可以被回收覆盖了
                                # 放在 finally 中确保即使读取报错，内存池坑位也不会永久丢失
                                result_queue.put_nowait(("FREE_BLOCK", shm_idx))


                            # ================= 阶段 C：Hailo NPU 推理 =================
                            try:
                                # 空数据拦截
                                if vehicle_img is None or vehicle_img.size == 0:
                                    continue

                                # 1. y5fu 定位 (NPU)
                                bboxes, landmarks = detector(vehicle_img, det_pipe)
                                if len(bboxes) == 0: 
                                    continue
                                
                                best_landmarks = landmarks[0].astype(np.float32)

                                # 2. 几何透视变换 (CPU)
                                M = cv2.getPerspectiveTransform(best_landmarks, dst_pts)
                                warped_plate = cv2.warpPerspective(vehicle_img, M, (96, 32))
                                
                                # 3. litemodel 颜色分类 (NPU)
                                lite_out = classifier(warped_plate, cls_pipe)
                                logits = np.squeeze(lite_out)
                                
                                if np.isclose(np.sum(logits), 1.0) and np.all(logits >= 0):
                                    probs = logits
                                else:
                                    exp_logits = np.exp(logits - np.max(logits))
                                    probs = exp_logits / np.sum(exp_logits)
                                
                                color_idx = np.argmax(probs)
                                conf = float(probs[color_idx])

                                # 4. 坐标归一化与投递结果
                                h, w = vehicle_img.shape[:2]
                                rel_landmarks = best_landmarks / np.array([w, h], dtype=np.float32)
                                
                                if conf > 0.3:
                                    result_queue.put_nowait((track_id, colors[color_idx], conf, rel_landmarks))

                            except Exception as e:
                                logger.error(f"推理崩溃 TID={track_id}: {e}")
                                err_str = str(e).lower()
                                if "hailo" in str(type(e)).lower() or "infer" in err_str or "vstream" in err_str:
                                    logger.warning("检测到 Hailo 管道损坏，正在触发热重启机制...")
                                    break # 打破内层循环，进入外层循环重建管道

                except Exception as fatal_e:
                    logger.error(f"NPU 挂载失败，等待底层驱动重置: {fatal_e}")
                    time.sleep(1.5)
