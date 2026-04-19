#!/bin/bash
# launch.sh

# 强杀所有之前的 Python 实例
pkill -9 -f app_gui.py
pkill -9 -f monitor_engine.py

# 稍微等一等，让 Linux 内核释放 /dev/i2c-1 和 /dev/hailo0 等硬件句柄
sleep 1.5

# 1. 切换工作目录到项目根目录，确保相对路径资源加载正常
cd /home/m0onck/traffic-emission-monitor-edge

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 配置环境变量
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export WAYLAND_DISPLAY=wayland-0
export QT_QPA_PLATFORM=wayland

# 确保项目根目录下的 bin 文件夹存在
mkdir -p bin

# ==========================================
# 4. 编译 C/C++ 硬件驱动动态库
# ==========================================

# 4.1 编译热成像动态库
echo "Building MLX90640 Thermal Camera Driver..."
make -C perception/sensor/mlx90640_driver clean
if make -C perception/sensor/mlx90640_driver libmlx90640.so; then
    mv perception/sensor/mlx90640_driver/libmlx90640.so ./bin/
    echo "MLX90640 Driver build complete. Artifact moved to ./bin/"
else
    echo "Error: MLX90640 Build failed!"
    exit 1
fi

# 4.2 编译气象站串口动态库
echo "Building Weather Station Driver..."
# 使用 g++ 直接编译为动态库。启用 -O3 优化，-fPIC 和 -shared 用于生成 .so 文件，-pthread 支持多线程
if g++ -O3 -Wall -fPIC -shared -pthread perception/sensor/weather_driver/libweather.cpp -o bin/libweather.so; then
    echo "Weather Station Driver build complete. Artifact generated at ./bin/libweather.so"
else
    echo "Error: Weather Station Driver build failed!"
    exit 1
fi

# ==========================================

# 强制开启底层 TRACE 级别日志，并输出到文件
export HAILORT_LOGGER_LEVEL=TRACE
export HAILORT_LOGGER_PATH=/home/m0onck/traffic-emission-monitor-edge/hailo_crash.log

# 5. 启动主程序
python app_gui.py
