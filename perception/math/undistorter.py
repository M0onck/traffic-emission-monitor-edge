# perception/math/undistorter.py

import cv2
import numpy as np

class CamUndistorter:
    def __init__(self, npz_path, resolution):
        """
        初始化定点数加速映射表
        :param npz_path: 标定文件路径 (resources/camera_calib_6mm.npz)
        :param resolution: 视频流分辨率 (width, height)，必须与初始化保持绝对一致
        """
        data = np.load(npz_path)
        mtx = data['mtx']
        dist = data['dist']
        w, h = resolution
        
        # 计算最佳新内参矩阵 (alpha=0 裁剪掉黑边，alpha=1 保留所有畸变像素并产生黑边)
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        
        # 核心优化：使用 cv2.CV_16SC2 极大地加速 ARM CPU 上的重映射过程
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            mtx, dist, None, new_mtx, (w, h), cv2.CV_16SC2
        )
        
    def process(self, frame):
        """
        执行极速映射
        """
        # cv2.INTER_LINEAR 是画质和速度的最佳平衡点
        return cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)
