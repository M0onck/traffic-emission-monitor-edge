import sys
from PyQt5.QtWidgets import QApplication
from multiprocessing import Process
import infra.config.loader as cfg

from ui.views.main_window import MainWindow
from ui.controllers.main_controller import MainController
from app.alignment_daemon import AlignmentDaemon

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
