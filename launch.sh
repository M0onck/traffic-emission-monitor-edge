#!/bin/bash
# launch.sh

# 1. 切换工作目录到项目根目录，确保相对路径资源加载正常
cd ~/traffic-emission-monitor-edge

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 配置显示环境变量（针对树莓派 GUI 环境的安全保险）
export DISPLAY=:0

# 4. 启动主程序
python3 app_gui.py
