import numpy as np
import cv2
from .base import HamburgerABC
from .multitask_detect import detect_pre_precessing, post_precessing, letter_box

# 引入 FormatType 强制声明硬件数据流类型
from hailo_platform import HEF, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType
import logging

logger = logging.getLogger(__name__)

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

        # 核心防线一：扩大硬件异步队列深度
        for stream_name in self.in_params.keys():
            self.in_params[stream_name].queue_size = 32
        for stream_name in self.out_params.keys():
            self.out_params[stream_name].queue_size = 32

        # 在类的生命周期内预分配一块唯一的、连续的物理内存
        # 杜绝 hailo_vdma_buffer_map 导致的内核 VMA 碎片化死锁
        # 兼容 3D (H, W, C) 和 4D (B, H, W, C) 的底层返回格式
        if len(self.input_shape) == 3:
            self.model_h, self.model_w, self.model_ch = self.input_shape
        elif len(self.input_shape) == 4:
            _, self.model_h, self.model_w, self.model_ch = self.input_shape
        else:
            raise ValueError(f"无法解析的输入形状: {self.input_shape}")
        self.static_input_buffer = np.ascontiguousarray(
            np.zeros((1, self.model_h, self.model_w, self.model_ch), dtype=np.float32)
        )
        
    def get_pipeline_args(self):
        return (self.network_group, self.in_params, self.out_params)

    def __call__(self, image, active_pipeline):
        frame_dict = self._preprocess(image)

        try:
            # 高阶 API infer() 会在底层走完 send -> recv
            raw_outputs = active_pipeline.infer(frame_dict)
        except Exception as e:
            raise RuntimeError(f"Classification 管道通信发生底层断流: {e}")

        array_out = raw_outputs[self.output_vstream_info.name].astype(np.float32)
        return self._postprocess(array_out)

    def _preprocess(self, image) -> dict:
        # cv2.resize 接受的参数格式是 (Width, Height)
        image_resize = cv2.resize(image, (self.model_w, self.model_h))
        image_rgb = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
        
        tensor = (image_rgb / 255.0).astype(np.float32)
        tensor = np.expand_dims(tensor, axis=0)
        
        # 使用 np.copyto 直接将数据刷入已映射好的物理地址，避免触发内核系统调用
        np.copyto(self.static_input_buffer, tensor)

        return {self.input_vstream_info.name: self.static_input_buffer}

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
        
        self.in_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.out_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

        # 扩大硬件异步队列深度
        for stream_name in self.in_params.keys():
            self.in_params[stream_name].queue_size = 32
        for stream_name in self.out_params.keys():
            self.out_params[stream_name].queue_size = 32

        # 零分配静态内存池
        self.input_shape = self.input_vstream_info.shape
        if len(self.input_shape) == 3:
            self.model_h, self.model_w, _ = self.input_shape
        elif len(self.input_shape) == 4:
            _, self.model_h, self.model_w, _ = self.input_shape
        else:
            raise ValueError(f"无法解析的输入形状: {self.input_shape}")
            
        self.input_size = (self.model_h, self.model_w)

        self.static_input_buffer = np.ascontiguousarray(
            np.zeros((1, self.model_h, self.model_w, 3), dtype=np.float32)
        )

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

        try:
            raw_outputs = active_pipeline.infer(frame_dict)
        except Exception as e:
            raise RuntimeError(f"Detector 管道通信发生底层断流: {e}")

        merged_output = self._decode_raw_logits(raw_outputs)
        return self._postprocess(merged_output)

    def _preprocess(self, image):
        img, r, left, top = letter_box(image, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        tensor = (img / 255.0).astype(np.float32)
        tensor = np.expand_dims(tensor, axis=0) 
        
        # 静态内存覆写
        np.copyto(self.static_input_buffer, tensor)
        
        self.tmp_pack = r, left, top
        return {self.input_vstream_info.name: self.static_input_buffer}

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
