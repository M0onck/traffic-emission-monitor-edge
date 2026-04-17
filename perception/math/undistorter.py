import cv2
import numpy as np

class CameraUndistorter:
    def __init__(self, npz_path):
        # 加载 .npz 文件
        data = np.load(npz_path)
        self.mtx = data['mtx']
        self.dist = data['dist']
        self.map_x = None
        self.map_y = None
        self.refined_mtx = None

    def init_maps(self, image_shape):
        """
        根据图像尺寸预计算映射表，只需要在第一帧调用一次
        image_shape: (height, width)
        """
        h, w = image_shape[:2]
        # 获取优化后的内参矩阵，alpha=0 表示保留所有有效像素
        self.refined_mtx, _ = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), 0, (w, h)
        )
        # 生成映射表
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.mtx, self.dist, None, self.refined_mtx, (w, h), cv2.CV_32FC1
        )

    def process_frame(self, frame):
        """对单帧图像进行去畸变"""
        if self.map_x is None:
            self.init_maps(frame.shape)
        # 使用重映射函数，速度极快
        return cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)
