# main.py
import faulthandler
faulthandler.enable()

import sys
import logging
# 配置全局日志基础设置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# 导入其他组件
from PyQt5.QtWidgets import QApplication
import multiprocessing as mp
import infra.config.loader as cfg

from app.bootstrap import AppBootstrap
from ui.views.main_window import MainWindow
from ui.controllers.main_controller import MainController

def main():
    # 强制使用 spawn 模式，避免 Linux 下 GUI 与多进程的资源死锁
    mp.set_start_method('spawn', force=True)

    from app.bootstrap import sync_native_extensions
    # 确保底层 C++ 库编译并注入环境变量
    native_lib_path = sync_native_extensions()
    # 对于传感器驱动，这里可以动态添加到加载路径，方便后面的 ctypes 调用
    sys.path.append(native_lib_path)

    # 启动主窗口
    app = QApplication(sys.argv)
    
    # 1. 调用工厂：一次性组装好所有的底层组件、队列和守护进程
    components = AppBootstrap.setup_components(cfg)

    # 2. 实例化 MVC 架构
    view = MainWindow()
    controller = MainController(view, components) 
    
    # 3. 显示界面并阻塞运行
    view.showFullScreen()
    exit_code = app.exec_()
    
    # 4. 安全退出与资源回收 (非常重要)
    print(">>> [System] 正在关闭系统并回收后台进程...")
    components['stop_event'].set() # 触发全局停止信号
    
    if components.get('alignment_proc') and components['alignment_proc'].is_alive():
        components['alignment_proc'].join(timeout=3)
        
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
