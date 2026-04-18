import numpy as np
import cv2
import os

# 确保使用绝对路径
base_dir = os.path.abspath(os.path.dirname(__file__))
npz_path = os.path.join(base_dir, 'camera_calib_6mm.npz')
bin_path = os.path.join(base_dir, 'dewarp_map_rgba_1456x1088.bin')

print(f"正在读取标定文件: {npz_path}")
calib_data = np.load(npz_path)
mtx, dist = calib_data['mtx'], calib_data['dist']
dim = (1456, 1088) 

# 生成 OpenCV 重映射表
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, dim, 0, dim)
map_x, map_y = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, dim, 5)

# 1. 坐标归一化并严格限制在 0.0 ~ 1.0 之间
map_x_norm = np.clip(map_x / dim[0], 0.0, 1.0)
map_y_norm = np.clip(map_y / dim[1], 0.0, 1.0)

# 2. 将 Float 映射到 16-bit 无符号整数 (0~65535)
x_int = (map_x_norm * 65535.0).astype(np.uint16)
y_int = (map_y_norm * 65535.0).astype(np.uint16)

# 3. 创建 RGBA 物理内存矩阵 (高 8 位和低 8 位分离)
map_rgba = np.zeros((dim[1], dim[0], 4), dtype=np.uint8)
map_rgba[..., 0] = (x_int >> 8) & 0xFF  # R 通道: X 的高位
map_rgba[..., 1] = x_int & 0xFF         # G 通道: X 的低位
map_rgba[..., 2] = (y_int >> 8) & 0xFF  # B 通道: Y 的高位
map_rgba[..., 3] = y_int & 0xFF         # A 通道: Y 的低位

# 4. 强制物理内存上下翻转并拷贝，完美契合 OpenGL 底层读取习惯
map_rgba = np.flipud(map_rgba).copy()

# 5. 写入磁盘
map_rgba.tofile(bin_path)

file_size = os.path.getsize(bin_path)
print(f"RGBA 映射表已生成！")
print(f"文件位置: {bin_path}")
print(f"文件大小: {file_size} bytes")
