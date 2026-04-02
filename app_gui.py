import sys
from PyQt5.QtWidgets import QApplication

from ui.views.main_window import MainWindow
from ui.controllers.main_controller import MainController

def main():
    app = QApplication(sys.argv)
    
    # 1. 实例化 View (视图)
    view = MainWindow()
    
    # 2. 实例化 Controller (控制器)，并将 View 注入
    controller = MainController(view)
    
    # 3. 显示界面
    view.showFullScreen()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
