#!/bin/bash
# launch.sh

# 1. 切换工作目录到项目根目录，确保相对路径资源加载正常
cd /home/m0onck/traffic-emission-monitor-edge

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 配置环境变量
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export WAYLAND_DISPLAY=wayland-0
export QT_QPA_PLATFORM=wayland

# 4. 启动主程序
python app_gui.py
