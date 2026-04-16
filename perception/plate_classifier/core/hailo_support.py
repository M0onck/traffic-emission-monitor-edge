import numpy as np
import cv2
from .base import HamburgerABC
from .multitask_detect import detect_pre_precessing, post_precessing, letter_box

class ClassificationHailo(HamburgerABC):
    """
    车牌属性分类模型 (Hailo NPU 硬件加速版)
    """
    def __init__(self, hef_path, target_vdevice, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from hailo_platform import HEF, InferVStreams, InputVStreamParams, OutputVStreamParams
        
        self.hef = HEF(hef_path)
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]
        self.input_shape = self.input_vstream_info.shape 
        
        self.network_group = target_vdevice.configure(self.hef)[0]
        self.in_params = InputVStreamParams.make(self.network_group)
        self.out_params = OutputVStreamParams.make(self.network_group)
        
    def get_pipeline_args(self):
        return (self.network_group, self.in_params, self.out_params)

    def __call__(self, image, active_pipeline):
        frame_dict = self._preprocess(image)
        raw_outputs = active_pipeline.infer(frame_dict)
        array_out = raw_outputs[self.output_vstream_info.name].astype(np.float32)
        return self._postprocess(array_out)

    def _preprocess(self, image) -> dict:
        image_resize = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        tensor = np.zeros((1, self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8)
        tensor[:] = image_resize 
        return {self.input_vstream_info.name: tensor}

    def _run_session(self, data: dict) -> np.ndarray:
        pass

    def _postprocess(self, data) -> np.ndarray:
        return data


class MultiTaskDetectorHailo(HamburgerABC):
    """
    车牌目标检测模型 (Hailo NPU 硬件加速版 - 计算图截断高精度解码版)
    """
    def __init__(self, hef_path, target_vdevice, box_threshold=0.5, nms_threshold=0.6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from hailo_platform import HEF, InputVStreamParams, OutputVStreamParams
        
        self.box_threshold = box_threshold
        self.nms_threshold = nms_threshold
        self.hef = HEF(hef_path)
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.input_size = (self.input_vstream_info.shape[0], self.input_vstream_info.shape[1])
        
        self.network_group = target_vdevice.configure(self.hef)[0]
        self.in_params = InputVStreamParams.make(self.network_group)
        self.out_params = OutputVStreamParams.make(self.network_group)

        # ====== 核心新增：YOLOv5 的解码参数 ======
        self.strides = [8, 16, 32]
        self.anchors = np.array([
            [[4, 5], [8, 10], [13, 16]],         # Stride 8
            [[23, 29], [43, 55], [73, 105]],     # Stride 16
            [[146, 217], [231, 300], [335, 433]] # Stride 32
        ], dtype=np.float32)

    def get_pipeline_args(self):
        return (self.network_group, self.in_params, self.out_params)

    def __call__(self, image, active_pipeline):
        frame_dict = self._preprocess(image)
        
        # 1. 拿到 NPU 的原始输出 (Raw Logits)
        raw_outputs = active_pipeline.infer(frame_dict)
        
        # 2. 调用 CPU 高精度解码器，将 Logits 还原为真实的像素坐标！
        merged_output = self._decode_raw_logits(raw_outputs)
        
        # 3. 送入 NMS 进行后处理
        return self._postprocess(merged_output)

    def _preprocess(self, image):
        img, r, left, top = letter_box(image, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = np.zeros((1, self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        tensor[:] = img 
        safe_tensor = np.ascontiguousarray(tensor)
        self.tmp_pack = r, left, top
        return {self.input_vstream_info.name: safe_tensor}

    def _decode_raw_logits(self, raw_outputs):
        """
        核心修复逻辑：在 CPU 端完成 YOLOv5 坐标系的 Sigmoid 与 锚点解码
        """
        # 将字典转为列表并按特征图面积(H*W)降序排列，确保顺序为大中小特征图 (Stride 8 -> 16 -> 32)
        arrays = list(raw_outputs.values())
        arrays.sort(key=lambda x: x.shape[1] * x.shape[2], reverse=True)
        
        decoded_outputs = []
        
        for i, stride in enumerate(self.strides):
            feat = arrays[i]
            batch, h, w, _ = feat.shape
            num_anchors = len(self.anchors[i])
            
            # 变形为 (Batch, H, W, Anchor数量, 15)
            feat = feat.reshape(batch, h, w, num_anchors, 15)
            
            # 生成网格坐标 (Grid)
            grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            grid = np.stack((grid_x, grid_y), axis=-1).reshape(1, h, w, 1, 2)
            anchor_grid = self.anchors[i].reshape(1, 1, 1, num_anchors, 2)
            
            # ====== 最关键的一步：执行高精度浮点 Sigmoid (消除量化误差) ======
            feat_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(feat, -50, 50)))
            
            # 解码边界框坐标 (x, y, w, h)
            xy = (feat_sigmoid[..., 0:2] * 2.0 - 0.5 + grid) * stride
            wh = (feat_sigmoid[..., 2:4] * 2.0) ** 2 * anchor_grid
            
            # 提取目标置信度和类别概率
            obj_conf = feat_sigmoid[..., 4:5]
            cls_probs = feat_sigmoid[..., 13:15]
            
            # 解码 4 个车牌关键点 (Landmarks)
            lmks = feat_sigmoid[..., 5:13]
            lmk_x = (lmks[..., 0::2] * 2.0 - 0.5 + grid[..., 0:1]) * stride
            lmk_y = (lmks[..., 1::2] * 2.0 - 0.5 + grid[..., 1:2]) * stride
            
            landmarks = np.empty_like(lmks)
            landmarks[..., 0::2] = lmk_x
            landmarks[..., 1::2] = lmk_y
            
            # 按原有后处理顺序拼合: [x,y,w,h, conf, lmks(8), cls(2)] -> 共 15 维
            decoded_feat = np.concatenate((xy, wh, obj_conf, landmarks, cls_probs), axis=-1)
            decoded_outputs.append(decoded_feat.reshape(batch, -1, 15))
            
        # 将三个尺度的预测结果拼接为 (1, N, 15) 的浮点矩阵
        return np.concatenate(decoded_outputs, axis=1).astype(np.float32)

    def _run_session(self, data):
        pass

    def _postprocess(self, data):
        r, left, top = self.tmp_pack
        output = post_precessing(data, r, left, top, self.box_threshold, self.nms_threshold)
        
        if len(output) == 0:
            return [], []
            
        bboxes = output[:, :5]
        landmarks = output[:, 5:13].reshape(-1, 4, 2)
        
        return bboxes, landmarks
