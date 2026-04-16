import numpy as np
import cv2
from .base import HamburgerABC
from .multitask_detect import detect_pre_precessing, post_precessing, letter_box

# 引入 FormatType 强制声明硬件数据流类型
from hailo_platform import HEF, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType

class ClassificationHailo(HamburgerABC):
    """车牌属性分类模型"""
    def __init__(self, hef_path, target_vdevice, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hef = HEF(hef_path)
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]
        self.input_shape = self.input_vstream_info.shape 
        
        self.network_group = target_vdevice.configure(self.hef)[0]
        
        # 强制声明输入输出为 FLOAT32
        self.in_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.out_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        
    def get_pipeline_args(self):
        return (self.network_group, self.in_params, self.out_params)

    def __call__(self, image, active_pipeline):
        frame_dict = self._preprocess(image)
        raw_outputs = active_pipeline.infer(frame_dict)
        array_out = raw_outputs[self.output_vstream_info.name].astype(np.float32)
        return self._postprocess(array_out)

    def _preprocess(self, image) -> dict:
        image_resize = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        image_rgb = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
        
        # 归一化处理并指定 np.float32
        tensor = (image_rgb / 255.0).astype(np.float32)
        tensor = np.expand_dims(tensor, axis=0) 
        safe_tensor = np.ascontiguousarray(tensor)
        return {self.input_vstream_info.name: safe_tensor}

    def _run_session(self, data: dict) -> np.ndarray: pass
    def _postprocess(self, data) -> np.ndarray: return data


class MultiTaskDetectorHailo(HamburgerABC):
    """车牌目标检测模型"""
    def __init__(self, hef_path, target_vdevice, box_threshold=0.5, nms_threshold=0.6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_threshold = box_threshold
        self.nms_threshold = nms_threshold
        self.hef = HEF(hef_path)
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.input_size = (self.input_vstream_info.shape[0], self.input_vstream_info.shape[1])
        
        self.network_group = target_vdevice.configure(self.hef)[0]
        
        # 强制声明 FLOAT32
        self.in_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.out_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

        self.strides = [8, 16, 32]
        self.anchors = np.array([
            [[4, 5], [8, 10], [13, 16]],         
            [[23, 29], [43, 55], [73, 105]],     
            [[146, 217], [231, 300], [335, 433]] 
        ], dtype=np.float32)

    def get_pipeline_args(self):
        return (self.network_group, self.in_params, self.out_params)

    def __call__(self, image, active_pipeline):
        frame_dict = self._preprocess(image)
        raw_outputs = active_pipeline.infer(frame_dict)
        merged_output = self._decode_raw_logits(raw_outputs)
        return self._postprocess(merged_output)

    def _preprocess(self, image):
        img, r, left, top = letter_box(image, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # ⚠️ 核心修复三：除以 255 并转为 FLOAT32
        tensor = (img / 255.0).astype(np.float32)
        tensor = np.expand_dims(tensor, axis=0) 
        safe_tensor = np.ascontiguousarray(tensor)
        
        self.tmp_pack = r, left, top
        return {self.input_vstream_info.name: safe_tensor}

    def _decode_raw_logits(self, raw_outputs):
        arrays = list(raw_outputs.values())
        arrays.sort(key=lambda x: x.shape[1] * x.shape[2], reverse=True)
        decoded_outputs = []
        
        for i, stride in enumerate(self.strides):
            feat = arrays[i]
            batch, h, w, _ = feat.shape
            num_anchors = len(self.anchors[i])
            
            feat = feat.reshape(batch, h, w, num_anchors, 15)
            grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            grid = np.stack((grid_x, grid_y), axis=-1).reshape(1, h, w, 1, 2)
            anchor_grid = self.anchors[i].reshape(1, 1, 1, num_anchors, 2)
            
            feat_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(feat, -50, 50)))
            
            xy = (feat_sigmoid[..., 0:2] * 2.0 - 0.5 + grid) * stride
            wh = (feat_sigmoid[..., 2:4] * 2.0) ** 2 * anchor_grid
            obj_conf = feat_sigmoid[..., 4:5]
            cls_probs = feat_sigmoid[..., 13:15]
            
            lmks = feat_sigmoid[..., 5:13]
            lmk_x = (lmks[..., 0::2] * 2.0 - 0.5 + grid[..., 0:1]) * stride
            lmk_y = (lmks[..., 1::2] * 2.0 - 0.5 + grid[..., 1:2]) * stride
            
            landmarks = np.empty_like(lmks)
            landmarks[..., 0::2] = lmk_x
            landmarks[..., 1::2] = lmk_y
            
            decoded_feat = np.concatenate((xy, wh, obj_conf, landmarks, cls_probs), axis=-1)
            decoded_outputs.append(decoded_feat.reshape(batch, -1, 15))
            
        return np.concatenate(decoded_outputs, axis=1).astype(np.float32)

    def _run_session(self, data): pass

    def _postprocess(self, data):
        r, left, top = self.tmp_pack
        output = post_precessing(data, r, left, top, self.box_threshold, self.nms_threshold)
        if len(output) == 0: return [], []
        bboxes = output[:, :5]
        landmarks = output[:, 5:13].reshape(-1, 4, 2)
        return bboxes, landmarks
