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
        # 导入 InferVStreams 和必备的流参数构造器
        from hailo_platform import HEF, InferVStreams, InputVStreamParams, OutputVStreamParams
        
        self.hef = HEF(hef_path)
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]
        self.input_shape = self.input_vstream_info.shape # 期望 (96, 96, 3)
        
        # 配置到 NPU
        self.network_group = target_vdevice.configure(self.hef)[0]
        
        # 显式创建输入和输出虚拟流参数
        self.in_params = InputVStreamParams.make(self.network_group)
        self.out_params = OutputVStreamParams.make(self.network_group)
        
    def get_pipeline_args(self):
        return (self.network_group, self.in_params, self.out_params)

    def __call__(self, image, active_pipeline):
        frame_dict = self._preprocess(image)
        
        # 直接传字典
        raw_outputs = active_pipeline.infer(frame_dict)
        
        # 提取结果并转换为float32
        array_out = raw_outputs[self.output_vstream_info.name].astype(np.float32)
        
        return self._postprocess(array_out)

    def _preprocess(self, image) -> dict:
        image_resize = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        
        tensor = np.zeros((1, self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8)
        
        tensor[:] = image_resize 
        
        return {self.input_vstream_info.name: tensor}

    def _run_session(self, data: dict) -> np.ndarray:
        # NPU 硬件推理
        result_dict = self.infer_pipeline.infer(data)
        return result_dict[self.output_vstream_info.name]

    def _postprocess(self, data) -> np.ndarray:
        return data


class MultiTaskDetectorHailo(HamburgerABC):
    """
    车牌目标检测模型 (Hailo NPU 硬件加速版 - 计算图截断高精度解码)
    """
    def __init__(self, hef_path, target_vdevice, box_threshold=0.5, nms_threshold=0.6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from hailo_platform import HEF, InputVStreamParams, OutputVStreamParams
        
        self.box_threshold = box_threshold
        self.nms_threshold = nms_threshold
        self.hef = HEF(hef_path)
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.input_size = (self.input_vstream_info.shape[0], self.input_vstream_info.shape[1])
        
        # 配置到 NPU
        self.network_group = target_vdevice.configure(self.hef)[0]
        self.in_params = InputVStreamParams.make(self.network_group)
        self.out_params = OutputVStreamParams.make(self.network_group)

        # y5fu (YOLOv5-Face) 标准 320x320 锚框配置
        # 若您训练时更改了 anchor，请在此同步修改
        self.strides = [8, 16, 32]
        self.anchors = np.array([
            [[4, 5], [8, 10], [13, 16]],         # P3 / Stride 8
            [[23, 29], [43, 55], [73, 105]],     # P4 / Stride 16
            [[146, 217], [231, 300], [335, 433]] # P5 / Stride 32
        ], dtype=np.float32)

    def get_pipeline_args(self):
        return (self.network_group, self.in_params, self.out_params)

    def __call__(self, image, active_pipeline):
        frame_dict = self._preprocess(image)
        
        # 1. NPU 执行推理，拿到 3 个未经过 Sigmoid 解码的原始特征图
        raw_outputs = active_pipeline.infer(frame_dict)
        
        # 2. CPU 高精度解码 (消除 NPU 量化误差)
        merged_output = self._decode_raw_logits(raw_outputs)
        
        # 3. 传入传统的 NMS 后处理阶段
        return self._postprocess(merged_output)

    def _preprocess(self, image):
        # 1. Letterbox 缩放
        img, r, left, top = letter_box(image, self.input_size)
        
        # 2. 转为 RGB 格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        tensor = np.zeros((1, self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        tensor[:] = img
        
        safe_tensor = np.ascontiguousarray(tensor)
        self.tmp_pack = r, left, top
        return {self.input_vstream_info.name: safe_tensor}

    def _run_session(self, data):
        """
        仅为满足 HamburgerABC 的抽象方法契约。
        在 Hailo 异步管线中，实际的硬件推理已在 __call__ 中被 active_pipeline.infer() 直接接管。
        """
        pass

    def _decode_raw_logits(self, raw_outputs):
        """
        核心修复逻辑：在 CPU 端完成 YOLOv5 坐标系的 Sigmoid 与 锚点解码
        """
        # 将 NPU 字典输出转换为列表，并按照物理特征图尺寸进行排序
        # 排序后顺序固定为：大特征图(小目标/Stride 8) -> 中等(Stride 16) -> 小特征图(大目标/Stride 32)
        arrays = list(raw_outputs.values())
        arrays.sort(key=lambda x: x.shape[1] * x.shape[2], reverse=True)
        
        decoded_outputs = []
        
        for i, stride in enumerate(self.strides):
            feat = arrays[i]
            batch, h, w, _ = feat.shape
            num_anchors = len(self.anchors[i])
            
            # 变形为 (Batch, H, W, Anchor数量, 属性维度15)
            # 15 = 4(框) + 1(框置信) + 8(四点坐标) + 2(类别概率)
            feat = feat.reshape(batch, h, w, num_anchors, 15)
            
            # 构建网格 (Grid)
            grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            grid = np.stack((grid_x, grid_y), axis=-1).reshape(1, h, w, 1, 2)
            
            anchor_grid = self.anchors[i].reshape(1, 1, 1, num_anchors, 2)
            
            # 执行高精度浮点 Sigmoid
            # 使用 clip 防止指数爆炸报警
            feat_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(feat, -50, 50)))
            
            # 1. 解码边界框坐标 (x, y, w, h)
            xy = (feat_sigmoid[..., 0:2] * 2.0 - 0.5 + grid) * stride
            wh = (feat_sigmoid[..., 2:4] * 2.0) ** 2 * anchor_grid
            
            # 2. 提取目标置信度与类别概率
            obj_conf = feat_sigmoid[..., 4:5]
            cls_probs = feat_sigmoid[..., 13:15]
            
            # 3. 解码车牌的 4 个关键点坐标 (x1, y1, x2, y2, x3, y3, x4, y4)
            lmks = feat_sigmoid[..., 5:13]
            lmk_x = (lmks[..., 0::2] * 2.0 - 0.5 + grid[..., 0:1]) * stride
            lmk_y = (lmks[..., 1::2] * 2.0 - 0.5 + grid[..., 1:2]) * stride
            
            landmarks = np.empty_like(lmks)
            landmarks[..., 0::2] = lmk_x
            landmarks[..., 1::2] = lmk_y
            
            # 按照原有后处理要求的顺序拼合
            decoded_feat = np.concatenate((xy, wh, obj_conf, landmarks, cls_probs), axis=-1)
            
            # 展平该尺度的所有预测结果，放入列表
            decoded_outputs.append(decoded_feat.reshape(batch, -1, 15))
            
        # 将三个尺度的预测结果横向拼接，输出完整的 (1, N, 15) 预测矩阵
        return np.concatenate(decoded_outputs, axis=1).astype(np.float32)

    def _postprocess(self, data):
        r, left, top = self.tmp_pack
        # 调用 multitask_detect.py 中的原始非极大值抑制逻辑
        output = post_precessing(data, r, left, top, self.box_threshold, self.nms_threshold)
        
        if len(output) == 0:
            return [], []
            
        bboxes = output[:, :5]
        landmarks = output[:, 5:13].reshape(-1, 4, 2)
        
        return bboxes, landmarks
