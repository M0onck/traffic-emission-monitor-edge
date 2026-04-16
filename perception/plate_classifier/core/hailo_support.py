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
        
        # 提取结果
        array_out = raw_outputs[self.output_vstream_info.name]
        
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
    车牌目标检测模型 (Hailo NPU 硬件加速版)
    """
    def __init__(self, hef_path, target_vdevice, box_threshold=0.5, nms_threshold=0.6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from hailo_platform import HEF, InferVStreams, InputVStreamParams, OutputVStreamParams
        
        self.box_threshold = box_threshold
        self.nms_threshold = nms_threshold
        self.hef = HEF(hef_path)
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.input_size = (self.input_vstream_info.shape[0], self.input_vstream_info.shape[1])
        
        # 配置到 NPU
        self.network_group = target_vdevice.configure(self.hef)[0]
        self.in_params = InputVStreamParams.make(self.network_group)
        self.out_params = OutputVStreamParams.make(self.network_group)

    def get_pipeline_args(self):
        return (self.network_group, self.in_params, self.out_params)

    def __call__(self, image, active_pipeline):
        frame_dict = self._preprocess(image)
        raw_outputs = active_pipeline.infer(frame_dict)
        
        target_suffixes = ['Concat_617', 'Concat_521', 'Concat_713']
        reshaped_outs = []
        
        for suffix in target_suffixes:
            matching_key = next((k for k in raw_outputs.keys() if suffix in k), None)
            if matching_key is None:
                raise KeyError(f"NPU 输出中找不到预期的特征图: {suffix}")
                
            flat_tensor = raw_outputs[matching_key].reshape(1, -1, 15) 
            reshaped_outs.append(flat_tensor)
        
        merged_output = np.concatenate(reshaped_outs, axis=1)
        return self._postprocess(merged_output)

    def _preprocess(self, image):
        # 1. Letterbox 缩放
        img, r, left, top = letter_box(image, self.input_size)
        
        # 2. 转为 RGB 格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        tensor = np.zeros((1, self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        tensor[:] = img # 将图像数据安全拷贝进容器
        
        # 强制申请一块绝对干净、连续的物理内存，防止它是 view（视图）
        safe_tensor = np.ascontiguousarray(tensor)
        
        self.tmp_pack = r, left, top
        return {self.input_vstream_info.name: safe_tensor}

    def _run_session(self, data):
        # NPU 硬件推理，返回的是 3 个被切断的特征图
        raw_outputs = self.infer_pipeline.infer(data)
        
        # ==========================================================
        # CPU 智能缝合逻辑：动态寻找 Key，并按 ONNX 的原本顺序拼接
        # ==========================================================
        # 这里的顺序必须和之前 ONNX 尾部的 Concat 顺序一致
        target_suffixes = ['Concat_617', 'Concat_521', 'Concat_713']
        reshaped_outs = []
        
        for suffix in target_suffixes:
            # 在返回的字典中寻找对应的完整键名 (例如: 'y5fu/Concat_617')
            matching_key = next((k for k in raw_outputs.keys() if suffix in k), None)
            if matching_key is None:
                raise KeyError(f"NPU 输出中找不到预期的特征图: {suffix}")
                
            # 将输出从例如 (1, 40, 40, 15) 拉平为 (1, 1600, 15)
            # 最后一个维度 15 表示 (x, y, w, h, score, + 10个关键点/其他信息)
            flat_tensor = raw_outputs[matching_key].reshape(1, -1, 15)
            reshaped_outs.append(flat_tensor)
        
        # 沿着锚框数量的维度(axis=1)拼接起来，完美伪装成原本 ONNX 的输出格式！
        merged_output = np.concatenate(reshaped_outs, axis=1)
        return merged_output

    def _postprocess(self, data):
        # 无缝对接原有的后处理逻辑
        r, left, top = self.tmp_pack
        output = post_precessing(data, r, left, top)
        
        # 修复逻辑：如果没有检测到目标，安全返回
        if len(output) == 0:
            return [], []
            
        bboxes = output[:, :5]
        landmarks = output[:, 5:13].reshape(-1, 4, 2)
        
        return bboxes, landmarks
