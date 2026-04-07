#!/bin/bash
# launch.sh

# 1. 切换工作目录到项目根目录，确保相对路径资源加载正常
cd /home/m0onck/traffic-emission-monitor-edge

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 启动主程序
python app_gui.py
