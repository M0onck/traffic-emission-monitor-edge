#!/bin/bash
# launch.sh

# 强杀所有之前的 Python 实例
pkill -9 -f app_gui.py
pkill -9 -f monitor_engine.py

# 稍微等一等，让 Linux 内核释放 /dev/i2c-1 和 /dev/hailo0 的硬件句柄
sleep 1.5

# 1. 切换工作目录到项目根目录，确保相对路径资源加载正常
cd /home/m0onck/traffic-emission-monitor-edge

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 配置环境变量
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export WAYLAND_DISPLAY=wayland-0
export QT_QPA_PLATFORM=wayland

# 4. 编译热成像动态库
echo "Building MLX90640 Thermal Camera Driver..."

# 确保项目根目录下的 bin 文件夹存在
mkdir -p bin

# 进入驱动目录执行清理和编译
# 使用 && 确保前一步成功才执行下一步
make -C perception/sensor/mlx90640_driver clean
if make -C perception/sensor/mlx90640_driver libmlx90640.so; then
    # 编译成功后，将生成的 .so 文件移动到 bin 目录
    mv perception/sensor/mlx90640_driver/libmlx90640.so ./bin/
    echo "Driver build complete. Artifact moved to ./bin/"
else
    echo "Error: Build failed!"
    exit 1
fi

# 强制开启底层 TRACE 级别日志，并输出到文件
export HAILORT_LOGGER_LEVEL=TRACE
export HAILORT_LOGGER_PATH=/home/m0onck/traffic-emission-monitor-edge/hailo_crash.log

# 5. 启动主程序
python app_gui.py
