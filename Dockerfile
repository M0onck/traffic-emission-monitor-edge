# 树莓派 5 (ARM64) 基础镜像
FROM ubuntu:24.04

# 防止 apt 安装时卡在交互提示
ENV DEBIAN_FRONTEND=noninteractive

# 设定工作目录
WORKDIR /app

# 安装系统级依赖 (编译工具链、GStreamer 框架、I2C/V4L2 硬件工具)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    libglib2.0-dev \
    libgl1-mesa-dev \
    i2c-tools \
    v4l-utils \
    gnupg \
    software-properties-common \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 导入 Hailo 官方 Apt 仓库并安装容器内依赖
# 这会提供 libhailort.so 以及 python3-hailort (即代码中 import hailo_platform 的来源)
RUN wget -O - https://hailo.ai/apt/hailo.gpg | apt-key add - && \
    echo "deb [arch=arm64] https://hailo.ai/apt/ stable main" | tee /etc/apt/sources.list.d/hailo.list && \
    apt-get update && apt-get install -y hailort python3-hailort \
    && rm -rf /var/lib/apt/lists/*

# 建立 Python 虚拟环境 (遵循 PEP 668 规范)
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
