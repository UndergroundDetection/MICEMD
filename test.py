# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 10:02:54 2020

@author: Administrator
"""

"""测试主界面"""
# import sys
# import numpy as np

# from PyQt5.QtWidgets import QApplication, QMainWindow
# from PyQt5.QtCore import QTranslator

# from matplotlib.figure import Figure
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# from mainwindow_ui import Ui_MainWindow
# from fdem.fdem_forward_simulation import Detector, Target, Collection
# from fdem.fdem_forward_simulation import fdem_forward_simulation
# from show import show_fdem_mag_map, show_discretize


# class MainWindow(QMainWindow, Ui_MainWindow):
#     """

#     """

#     def __init__(self):

#         super(MainWindow, self).__init__()
#         self.setupUi(self)

#         desktop = QApplication.desktop()
#         x = (desktop.width() - self.width()) // 2
#         y = (desktop.height()-65 - self.height()) // 2
#         self.move(x, y)

#         self.trans = QTranslator(self)

#         self.select_detection_method()

#         self.fig_discretize = Figure(figsize=(4.21, 3.91))
#         self.canvas_discretize = FigureCanvasQTAgg(self.fig_discretize)
#         self.gl_discretize.addWidget(self.canvas_discretize)

#         self.fig_scenario = Figure(figsize=(4.21, 3.91))
#         self.canvas_scenario = FigureCanvasQTAgg(self.fig_scenario)
#         self.gl_detection_scenario.addWidget(self.canvas_scenario)

#         self.fig_magnetic_field = Figure(figsize=(4.21, 3.91))
#         self.canvas_magnetic_field = FigureCanvasQTAgg(self.fig_magnetic_field)
#         self.gl_magnetic_field_map.addWidget(self.canvas_magnetic_field)


#         # connect slot
#         self.tab_signal_type.currentChanged.connect(
#             self.select_detection_method)
#         self.pb_run_fdem_forward_simulation.clicked.connect(
#             self.run_fdem_forward_simulation)
#         self.le_detector_radius.editingFinished.connect(self.get_fdem_simulation_parameters)

#     def select_detection_method(self):
#         """


#         Returns
#         -------
#         None.

#         """
#         if self.tab_signal_type.currentWidget() == self.tab_FDEM:
#             self.pb_run_fdem_forward_simulation.setVisible(True)
#             self.pb_run_fdem_inversion.setVisible(True)
#             self.pb_run_tdem_forward_simulation.setVisible(False)
#             self.pb_run_tdem_classification.setVisible(False)

#             self.get_fdem_simulation_parameters()
#         else:
#             self.pb_run_fdem_forward_simulation.setVisible(False)
#             self.pb_run_fdem_inversion.setVisible(False)
#             self.pb_run_tdem_forward_simulation.setVisible(True)
#             self.pb_run_tdem_classification.setVisible(True)

#     def get_fdem_simulation_parameters(self):
#         """

#         Return
#         -------
#         None.

#         """

#         self.detector = Detector(
#             float(self.le_detector_radius.text()),
#             float(self.le_detector_current.text()),
#             float(self.le_detector_frequency.text()),
#             float(self.le_detector_pitch.text()),
#             float(self.le_detector_roll.text())
#             )
#         self.target = Target(
#             float(self.le_target_conductivity.text()),
#             float(self.le_target_permeability.text()),
#             float(self.le_target_radius.text()),
#             float(self.le_target_pitch.text()),
#             float(self.le_target_roll.text()),
#             float(self.le_target_length.text()),
#             float(self.le_target_position_x.text()),
#             float(self.le_target_position_y.text()),
#             float(self.le_target_position_z.text())
#             )
#         self.collection = Collection(
#             float(self.le_collection_spacing.text()),
#             float(self.le_collection_height.text()),
#             float(self.le_collection_SNR.text()),
#             float(self.le_collection_x_min.text()),
#             float(self.le_collection_x_max.text()),
#             float(self.le_collection_y_min.text()),
#             float(self.le_collection_y_max.text()),
#             self.cb_collection_direction.currentText()
#             )

#         self.optimization_algorithm = \
#             self.cb_optimization_algorithm.currentText()
#         self.optimization_iterations = \
#             float(self.le_optimization_iterations.text())
#         self.optimization_tol = float(self.le_optimization_tol.text())

#         # self.result = Result()
#         # result.

#     def run_fdem_forward_simulation(self):
#         self.tb_output_box.setText(str(self.detector.radius)+' '+self.le_detector_radius.text())

# if __name__ == "__main__":

#     app = QApplication(sys.argv)
#     app.setStyle('Fusion')

#     m = MainWindow()
#     m.show()

#     sys.exit(app.exec_())

"""测试textbox显示"""
# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow

# from mainwindow_ui import Ui_MainWindow
# from PyQt5.QtWidgets import *
# class MainWindow(QMainWindow, Ui_MainWindow):
    
#     def __init__(self):
        
#         super(MainWindow, self).__init__()
#         self.setupUi(self)
        
#         desktop = QApplication.desktop()
#         x = (desktop.width() - self.width()) // 2
#         y = (desktop.height()-65 - self.height()) // 2
#         self.move(x, y)
        
#         self.x = 5
        
#         mytest(self.tb_output_box)
#         self.testvar()
        
#         self.le_detector_radius.textEdited.connect(self.myshow)
        
#     def myshow(self):
#         self.tb_output_box.setText(self.le_detector_radius.displayText())
        
#     def testvar(self):
#         x = 7
#         print(self.x)
        

# def mytest(output):
#     output.setText("yes")

# if __name__ == "__main__":
    
#     app = QApplication(sys.argv)
#     app.setStyle('Fusion')
    
#     m = MainWindow()
#     m.show()
    
#     sys.exit(app.exec_())

"""测试画图"""
# import matplotlib
# # matplotlib.use('Qt5Agg')
# # 使用 matplotlib中的FigureCanvas (在使用 Qt5 Backends中 FigureCanvas继承自QtWidgets.QWidget)
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from PyQt5 import QtCore, QtWidgets,QtGui
# import matplotlib.pyplot as plt
# from matplotlib.figure import Figure
# import sys

# class My_Main_window(QtWidgets.QDialog):
#     def __init__(self,parent=None):
#         # 父类初始化方法
#         super(My_Main_window,self).__init__(parent)
        
#         # 几个QWidgets
#         self.figure = Figure()
#         self.canvas = FigureCanvas(self.figure)
#         self.button_plot = QtWidgets.QPushButton("绘制")

#         # 连接事件
#         self.button_plot.clicked.connect(self.plot_)
        
#         # 设置布局
#         layout = QtWidgets.QVBoxLayout()
#         layout.addWidget(self.canvas)
#         layout.addWidget(self.button_plot)
#         self.setLayout(layout)

#     # 连接的绘制的方法
#     def plot_(self):
#         self.figure.clf()
#         ax = self.figure.add_axes([0.1,0.1,0.8,0.8])
#         ax.plot([1,2,3,4,5])
#         ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
#         self.canvas.draw()

# # 运行程序
# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     main_window = My_Main_window()
#     main_window.show()
#     app.exec()

"""测试多线程"""
 
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
 
global sec
sec=0
 
class WorkThread(QThread):
    trigger = pyqtSignal()
    def __int__(self):
        super(WorkThread,self).__init__()
        self.x = 5
 
    def run(self):
        for i in range(203300030):
            pass
        self.trigger.emit()         #循环完毕后发出信号
 
def countTime():
    global  sec
    sec+=1
    lcdNumber.display(sec)          #LED显示数字+1
 
def work():
    timer.start(1000)               #计时器每秒计数
    workThread=WorkThread()
    workThread.trigger.connect(timeStop)   #当获得循环完毕的信号时，停止计数
    workThread.start()              #计时开始
    print('main id', int(QThread.currentThreadId()))
    workThread.x = 6
    print(workThread.x)
    workThread.x = 7
    print(workThread.x)
 
def timeStop():
    timer.stop()
    print("运行结束用时",lcdNumber.value())
    global sec
    sec=0
 
app=QApplication([])
top=QWidget()
layout=QVBoxLayout(top)             #垂直布局类QVBoxLayout；
lcdNumber=QLCDNumber()              #加个显示屏
layout.addWidget(lcdNumber)
button=QPushButton("测试")
layout.addWidget(button)
 
timer=QTimer()

 
button.clicked.connect(work)
timer.timeout.connect(countTime)      #每次计时结束，触发setTime
 
top.show()
app.exec()
