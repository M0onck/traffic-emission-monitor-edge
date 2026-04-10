import cv2
import numpy as np
from .base import HamburgerABC


def encode_images(image: np.ndarray):
    image_encode = image / 255.0
    if len(image_encode.shape) == 4:
        image_encode = image_encode.transpose(0, 3, 1, 2)
    else:
        image_encode = image_encode.transpose(2, 0, 1)
    image_encode = image_encode.astype(np.float32)

    return image_encode


class ClassificationORT(HamburgerABC):

    def __init__(self, onnx_path, *args, **kwargs):
        import onnxruntime as ort
        super().__init__(*args, **kwargs)
        # 配置 ONNX Runtime 日志级别
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # 3 表示只输出 Error 及以上级别的日志

        # 将 sess_options 传入 InferenceSession
        self.session = ort.InferenceSession(onnx_path, sess_options)
        self.input_config = self.session.get_inputs()[0]
        self.output_config = self.session.get_outputs()[0]
        self.input_size = tuple(self.input_config.shape[2:])

    # @cost('Cls')
    def _run_session(self, data) -> np.ndarray:
        result = self.session.run([self.output_config.name], {self.input_config.name: data})

        return result[0]

    def _postprocess(self, data) -> np.ndarray:
        return data

    def _preprocess(self, image) -> np.ndarray:
        assert len(
            image.shape) == 3, "The dimensions of the input image object do not match. The input supports a single " \
                               "image. "
        # print(self.input_size)
        image_resize = cv2.resize(image, self.input_size)
        encode = encode_images(image_resize)
        encode = encode.astype(np.float32)
        input_tensor = np.expand_dims(encode, 0)

        return input_tensor


