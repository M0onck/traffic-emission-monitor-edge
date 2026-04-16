import cv2
import numpy as np
import multiprocessing as mp
import queue
from perception.plate_classifier.core.multitask_detect import letter_box, post_precessing
from perception.plate_classifier.core.hailo_support import MultiTaskDetectorHailo, ClassificationHailo

class AsyncPlateRecognizer:
    def __init__(self, y5fu_onnx_path, litemodel_onnx_path, num_workers=1):
        ctx = mp.get_context('spawn')
        self.task_queue = ctx.Queue(maxsize=30)
        self.result_queue = ctx.Queue()
        self.workers = [] # 保存子进程的引用

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
        if not self.task_queue.full():
            try:
                self.task_queue.put_nowait((track_id, np.ascontiguousarray(vehicle_crop)))
                return True
            except queue.Full:
                return False
        return False

    def get_results(self):
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def stop(self):
        """强制终止所有的 OCR 工作进程"""
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.0)
        print("[AsyncRecognizer] 所有 Hailo OCR 子进程已安全销毁。")

    @staticmethod
    def _worker_loop(task_queue, result_queue, y5fu_path, lite_path, worker_id):
        # 1. 引入 Hailo 虚拟设备管理器
        from hailo_platform import VDevice, InferVStreams, HailoSchedulingAlgorithm

        # 2. 创建允许跨进程共享的参数
        params = VDevice.create_params()
        params.group_id = "SHARED" 
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.multi_process_service = True # 启用多进程服务开关
        
        # 3. 在子进程内部接管 NPU 硬件资源
        with VDevice(params) as target_vdevice:
            print(f">>> [Worker {worker_id}] 已成功挂载 Hailo VDevice 并开始加载 HEF...")
            
            # 4. 实例化你封装好的 NPU 推理引擎
            detector = MultiTaskDetectorHailo(y5fu_path, target_vdevice)
            classifier = ClassificationHailo(lite_path, target_vdevice)

            det_args = detector.get_pipeline_args()
            cls_args = classifier.get_pipeline_args()

            colors = ["blue", "green", "yellow", "white", "black"]
            # 车牌标准几何透视点
            dst_pts = np.array([[0, 0], [96, 0], [96, 32], [0, 32]], dtype=np.float32)

            with InferVStreams(*det_args) as det_pipe, InferVStreams(*cls_args) as cls_pipe:
                print(f">>> [Worker {worker_id}] 所有的 NPU 推理通道已成功安全建立！")

                while True:
                    try:
                        track_id, vehicle_img = task_queue.get()
                    except Exception:
                        continue

                    try:
                        # ================= 1. y5fu 定位 (NPU) =================
                        # 直接调用实例，底层代码已处理好 Letterbox 缩放和特征图拼接
                        bboxes, landmarks = detector(vehicle_img, det_pipe)
                        
                        if len(bboxes) == 0: 
                            continue
                        
                        # 取置信度最高的第一个车牌关键点
                        best_landmarks = landmarks[0].astype(np.float32)

                        # ================= 2. 几何透视变换 (CPU) =================
                        # 利用 NPU 给出的坐标，在 CPU 执行几毫秒的轻量级拉直
                        M = cv2.getPerspectiveTransform(best_landmarks, dst_pts)
                        warped_plate = cv2.warpPerspective(vehicle_img, M, (96, 32))
                        
                        # ================= 3. litemodel 颜色分类 (NPU) =================
                        # 直接将拉直后的图片扔给分类器
                        lite_out = classifier(warped_plate, cls_pipe)
                        
                        # 剥离输出的 Batch 维度 (例如从 (1, 5) 变为 (5,))
                        logits = np.squeeze(lite_out)
                        
                        # 智能概率转换：兼容 Raw Logits 和 Pre-Softmax
                        if np.isclose(np.sum(logits), 1.0) and np.all(logits >= 0):
                            probs = logits
                        else:
                            exp_logits = np.exp(logits - np.max(logits))
                            probs = exp_logits / np.sum(exp_logits)
                        
                        color_idx = np.argmax(probs)
                        conf = float(probs[color_idx])

                        # ================= 4. 坐标归一化与投递 =================
                        h, w = vehicle_img.shape[:2]
                        # 归一化操作，生成 0.0 ~ 1.0 之间的相对坐标
                        rel_landmarks = best_landmarks / np.array([w, h], dtype=np.float32)
                        if conf > 0.3:
                            print(f"[DEBUG 2 子进程] NPU 识别出车牌! TID={track_id}, 颜色={colors[color_idx]}, 置信度={conf:.2f}")
                            result_queue.put_nowait((track_id, colors[color_idx], conf, rel_landmarks))
                        else:
                            print(f"[DEBUG 2 子进程] 识别失败 TID={track_id}, 置信度过低 ({conf:.2f} < 0.3)")

                    except Exception as e:
                        print(f"[ERROR 子进程异常] TID={track_id} 推理崩溃: {e}")
