#!/bin/bash

# ==============================================================================
# 树莓派 5 边缘监控节点 - 宿主机硬件初始化脚本
# ==============================================================================

# 遇到错误立即退出
set -e

# 1. 提权检查：确保脚本以 root 权限运行
if [ "$EUID" -ne 0 ]; then
  echo "[错误] 配置硬件层需要超级管理员权限。请使用 sudo ./setup_host.sh 运行。"
  exit 1
fi

echo "开始初始化 Traffic-Emission-Monitor 边缘硬件环境..."

# 获取脚本所在的绝对目录，确保后续路径寻找不会因为执行路径不同而出错
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
TEMPLATE_FILE="$SCRIPT_DIR/host_config_template.txt"
TARGET_FILE="/boot/firmware/config.txt"
BACKUP_FILE="/boot/firmware/config.txt.bak_$(date +%Y%m%d_%H%M%S)"

# 2. 检查模板文件是否存在
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "[错误] 找不到配置文件模板 $TEMPLATE_FILE"
    exit 1
fi

# 3. 安全机制：备份原有的 config.txt
echo "正在备份原始系统配置到 $BACKUP_FILE ..."
cp "$TARGET_FILE" "$BACKUP_FILE"

# 4. 覆写 config.txt
echo "正在应用定制化硬件拓扑与传感器配置..."
cp "$TEMPLATE_FILE" "$TARGET_FILE"

# 5. 权限配置：将常用非 root 用户 (默认通常是 pi 或自定义的用户) 加入硬件访问组
# 获取调用 sudo 的真实用户名
REAL_USER=${SUDO_USER:-$USER}
echo "正在为用户 '$REAL_USER' 配置 I2C 和 串口 (Dialout) 权限..."
usermod -aG i2c,dialout "$REAL_USER" || true

# 6. 系统工具链预装 (如果需要宿主机层面的驱动，如 HailoRT 基础依赖，可在此处追加)
echo "检查并安装基础宿主机依赖 (如 i2c-tools)..."
apt-get update -qq
apt-get install -y -qq i2c-tools rpicam-apps

# 7. 配置与启动底层 TCP 相机推流服务 (camera-feeder)
echo "[Host Setup] 创建并配置 TCP MJPEG 相机推流守护进程 (camera-feeder.service)..."
SERVICE_FILE="/etc/systemd/system/camera-feeder.service"

cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=TCP MJPEG Camera Server (rpicam-vid)
After=network.target

[Service]
Type=simple
User=$REAL_USER
# 使用原生 rpicam-vid 输出 1456x1088 30fps MJPEG 视频流，并监听本机 5000 端口
ExecStart=/usr/bin/rpicam-vid -t 0 --width 1456 --height 1088 --framerate 30 --codec mjpeg --listen -o tcp://0.0.0.0:5000
Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
EOF

# 重新加载 Systemd，设置开机自启并立刻启动服务
systemctl daemon-reload
systemctl enable camera-feeder.service
systemctl restart camera-feeder.service

echo "=============================================================================="
echo "宿主机硬件初始化完成！"
echo "变更需要重启才能生效，特别是 PCIe Gen 2、IMX296 的 3.3V PWM 约束以及 I2C 总线。"
echo "=============================================================================="

# 7. 优雅地提示重启
read -p "是否立即重启设备？(y/N): " REBOOT_CONFIRM
if [[ "$REBOOT_CONFIRM" =~ ^[Yy]$ ]]; then
    echo "系统即将重启..."
    reboot
else
    echo "稍后请手动执行 sudo reboot。"
fi
