import sys
import logging
from PyQt5.QtWidgets import QApplication
from multiprocessing import Process
import infra.config.loader as cfg

from ui.views.main_window import MainWindow
from ui.controllers.main_controller import MainController
from app.alignment_daemon import AlignmentDaemon

# 配置全局日志基础设置
logging.basicConfig(
    level=logging.INFO,  # 默认级别设为 INFO，这样 DEBUG 级别的信息默认不会打印
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout) # 输出到控制台
        # 如果需要输出到文件，可以加上：logging.FileHandler("edge_monitor.log")
    ]
)

# 车牌识别链路调试日志
# logging.getLogger("app.monitor_engine").setLevel(logging.DEBUG)

# GStreamer 硬件流媒体层调试日志
# logging.getLogger("perception.gst_pipeline").setLevel(logging.DEBUG)

def start_daemon():
    """启动独立进程运行 Daemon"""
    daemon = AlignmentDaemon(cfg)
    daemon.run()

def main():
    app = QApplication(sys.argv)
    
    daemon_process = None
    # 架构解耦：仅在 inference 模式下，于边缘端并行启动特征对齐守护进程
    if cfg.RUN_MODE == 'inference':
        daemon_process = Process(target=start_daemon, daemon=True)
        daemon_process.start()
    else:
        print(">>> [System] 当前为 collection 采集模式，后台对齐引擎已关闭。")

    # 1. 实例化 View (视图)
    view = MainWindow()
    
    # 2. 实例化 Controller (控制器)
    controller = MainController(view)
    
    # 3. 显示界面
    view.showFullScreen()
    
    # 阻塞运行 GUI
    exit_code = app.exec_()
    
    # 优雅退出守护进程
    if daemon_process and daemon_process.is_alive():
        daemon_process.terminate()
        daemon_process.join()

    sys.exit(exit_code)

if __name__ == '__main__':
    main()
