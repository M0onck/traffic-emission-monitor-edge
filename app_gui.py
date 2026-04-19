import os
# 强制约束底层 C++ 数学库的线程衍生，防止把树莓派的 CPU 抢爆
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import logging
import faulthandler
faulthandler.enable()
from PyQt5.QtWidgets import QApplication
from multiprocessing import Process, Event
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

def start_daemon(stop_event):
    """将事件对象传入守护进程"""
    from app.alignment_daemon import AlignmentDaemon
    daemon = AlignmentDaemon(cfg, stop_event) # 传入 stop_event
    daemon.run()

def main():
    app = QApplication(sys.argv)
    
    daemon_process = None
    stop_event = Event()
    # 仅在 inference 模式下，于边缘端并行启动特征对齐守护进程
    if cfg.RUN_MODE == 'inference':
        daemon_process = Process(target=start_daemon, args=(stop_event,), daemon=True)
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
    
    # 安全退出守护进程
    if daemon_process and daemon_process.is_alive():
        print(">>> [System] 正在发送守护进程退出信号...")
        stop_event.set() # 发送安全停止信号
        daemon_process.join(timeout=3.0) # 留给它 3 秒时间完成最后的数据库事务
        
        # 兜底：如果 3 秒还没退出（发生死锁），才使用 terminate
        if daemon_process.is_alive():
            print(">>> [Warning] 守护进程超时未退出，已强制退出.")
            daemon_process.terminate()

    sys.exit(exit_code)

if __name__ == '__main__':
    main()
