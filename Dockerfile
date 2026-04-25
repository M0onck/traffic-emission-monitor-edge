# 树莓派 5 (ARM64) 基础镜像
FROM debian:bookworm

# 防止 apt 安装时卡在交互提示
ENV DEBIAN_FRONTEND=noninteractive

# 设定工作目录
WORKDIR /app

# 提前安装网络与证书工具，用于获取 GPG 密钥
RUN apt-get update && apt-get install -y wget gnupg ca-certificates

# 导入 Raspberry Pi 官方源的安全密钥并配置源列表
RUN wget -qO - http://archive.raspberrypi.com/debian/raspberrypi.gpg.key | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] http://archive.raspberrypi.com/debian/ bookworm main" > /etc/apt/sources.list.d/raspi.list

# APT 优先级配置：让树莓派官方源的所有包优先级全面高于 Debian 官方源
# 解决未来由于底层关联依赖（如 libdrm, wayland 等）造成的版本冲突
RUN echo "Package: *" > /etc/apt/preferences.d/raspberrypi-pin && \
    echo "Pin: origin archive.raspberrypi.com" >> /etc/apt/preferences.d/raspberrypi-pin && \
    echo "Pin-Priority: 1001" >> /etc/apt/preferences.d/raspberrypi-pin

# 安装系统级依赖 (编译工具链、GStreamer 框架、I2C/V4L2 硬件工具、PyQt5/X11 组件)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    libxcb-cursor0 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libx11-xcb1 \
    libgl1 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-gl \
    libglib2.0-dev \
    libgl1-mesa-dev \
    libgl1-mesa-dri \
    libegl1-mesa \
    libgles2-mesa \
    i2c-tools \
    ffmpeg \
    gnupg \
    software-properties-common \
    wget \
    hailort \
    python3-hailort \
    && rm -rf /var/lib/apt/lists/*

# 安装基础依赖和系统级 PyQt5
RUN apt-get update && apt-get install -y \
    python3-pyqt5 \
    && rm -rf /var/lib/apt/lists/*

# 建立 Python 虚拟环境 (遵循 PEP 668 规范)
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 设置 pip 全局高速镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装 Python 依赖，需要额外删除其他库附带的其他版本 opencv-python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.9.0.80

# 拷贝项目全量源码
COPY . .

# 触发 C++ 扩展预编译 (MLX90640, Weather Station, Dewarp Plugin)
RUN python -c "from app.bootstrap import sync_native_extensions; sync_native_extensions()"

# 配置核心环境变量
# 确保 GStreamer 能找到 dewarpfilter.so，Python 能找到底层驱动 .so
ENV GST_PLUGIN_PATH=/app/build/lib:$GST_PLUGIN_PATH
ENV PYTHONPATH=/app/build/lib:$PYTHONPATH

# 启动程序
CMD ["python", "main.py"]
