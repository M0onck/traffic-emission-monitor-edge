import cv2
import numpy as np
import onnxruntime as ort
import multiprocessing as mp
import queue
from perception.plate_classifier.core.multitask_detect import letter_box, post_precessing

class AsyncPlateRecognizer:
    def __init__(self, y5fu_onnx_path, litemodel_onnx_path):
        ctx = mp.get_context('spawn')
        self.task_queue = ctx.Queue(maxsize=30)
        self.result_queue = ctx.Queue()

        print(">>> [AsyncRecognizer] 启动独立 Spawn 进程 ONNX 引擎...")

        self.worker_process = ctx.Process(
            target=self._worker_loop,
            args=(self.task_queue, self.result_queue, y5fu_onnx_path, litemodel_onnx_path),
            daemon=True
        )
        self.worker_process.start()

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

    @staticmethod
    def _worker_loop(task_queue, result_queue, y5fu_path, lite_path):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        
        sess_y5fu = ort.InferenceSession(y5fu_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
        sess_lite = ort.InferenceSession(lite_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

        colors = ["蓝牌", "绿牌", "黄牌", "白牌", "黑牌"]
        dst_pts = np.array([[0, 0], [96, 0], [96, 32], [0, 32]], dtype=np.float32)

        while True:
            try:
                track_id, vehicle_img = task_queue.get()
            except Exception:
                continue

            try:
                # ================= 1. y5fu 定位 =================
                img, r, left, top = letter_box(vehicle_img, (320, 320))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_chw = np.transpose(img_rgb, (2, 0, 1)) 
                tensor = np.expand_dims((img_chw / 255.0).astype(np.float32), axis=0)

                in_name = sess_y5fu.get_inputs()[0].name
                raw_outputs = sess_y5fu.run(None, {in_name: tensor})

                sorted_outputs = sorted(raw_outputs, key=lambda x: x.shape[2], reverse=True)
                reshaped_outputs = []
                for out in sorted_outputs:
                    if len(out.shape) == 4 and out.shape[1] == 15:
                        out = np.transpose(out, (0, 2, 3, 1))
                    reshaped_outputs.append(out.reshape(1, -1, 15))
                
                merged = np.concatenate(reshaped_outputs, axis=1)
                p_out = post_precessing(merged, r, left, top)

                if len(p_out) == 0: 
                    continue
                
                best_plate = p_out[0]
                landmarks = best_plate[5:13].reshape(4, 2).astype(np.float32)

                # ================= 2. 几何透视变换 =================
                M = cv2.getPerspectiveTransform(landmarks, dst_pts)
                warped_plate = cv2.warpPerspective(vehicle_img, M, (96, 32))
                
                # ================= 3. litemodel 颜色分类 =================
                plate_resize = cv2.resize(warped_plate, (96, 96))
                plate_chw = np.transpose(plate_resize, (2, 0, 1))
                lite_tensor = np.expand_dims((plate_chw / 255.0).astype(np.float32), axis=0)
                
                lite_in_name = sess_lite.get_inputs()[0].name
                lite_out = sess_lite.run(None, {lite_in_name: lite_tensor})
                
                logits = lite_out[0][0]
                
                # 🚨 智能概率转换：完美兼容 Raw Logits 和 Pre-Softmax
                if np.isclose(np.sum(logits), 1.0) and np.all(logits >= 0):
                    probs = logits
                else:
                    exp_logits = np.exp(logits - np.max(logits))
                    probs = exp_logits / np.sum(exp_logits)
                
                color_idx = np.argmax(probs)
                conf = float(probs[color_idx])

                # 🚨 降低子进程的拦截阈值，只要有一点倾向性就发回给主进程处理
                if conf > 0.3:
                    result_queue.put_nowait((track_id, colors[color_idx], conf))

            except Exception as e:
                # 静默捕获避免刷屏
                pass