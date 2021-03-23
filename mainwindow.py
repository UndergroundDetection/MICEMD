# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 10:02:54 2020

@author: Wang Zhen
"""

import sys

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTranslator

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from mainwindow_ui import Ui_MainWindow
from result import TFResult
from MicEMD.fdem.collection import Collection
from MicEMD.fdem.target import Target
from MicEMD.fdem.detector import Detector
from utilities.show import show_fdem_detection_scenario
from utilities.threadSet import ThreadCalFdem, ThreadInvFdem
from MicEMD.handler import FDEMHandler


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.initialize()
        # Connect slots function.
        self.get_fdem_simulation_parameters()
        self.connect_slots()
        self.thread_cal_fdem = ThreadCalFdem()
        self.thread_inv_fdem = ThreadInvFdem()

    def initialize(self):

        # Define the fdem forward simulation thread class.
        # self.thread_cal_fdem = ThreadCalFdem()
        # Define the fdem inversion thread class.
        #self.thread_inv_fdem = ThreadInvFdem()

        # Set the display position of the mainwindow.
        desktop = QApplication.desktop()
        x = (desktop.width() - self.width()) // 2
        y = (desktop.height() - 65 - self.height()) // 2
        self.move(x, y)

        # Desine the translator to translate interface languages.
        self.trans = QTranslator(self)
        # Define the Result class to record the results in the process.
        self.result = TFResult()

        # Define the figure to show data in the interface.
        self.fig_scenario = Figure(figsize=(4.21, 3.91))
        self.canvas_scenario = FigureCanvasQTAgg(self.fig_scenario)
        self.gl_detection_scenario.addWidget(self.canvas_scenario)

        self.fig_discretize = Figure(figsize=(4.21, 3.91))
        self.canvas_discretize = FigureCanvasQTAgg(self.fig_discretize)
        self.gl_discretize.addWidget(self.canvas_discretize)

        self.fig_magnetic_field = Figure(figsize=(4.21, 3.91))
        self.canvas_magnetic_field = FigureCanvasQTAgg(self.fig_magnetic_field)
        self.gl_magnetic_field_data.addWidget(self.canvas_magnetic_field)

        self.pbar_rfs.setVisible(False)
        self.pbar_rfi.setVisible(False)

    def connect_slots(self):

        self.actionChinese.triggered.connect(self.select_Chinese)
        self.actionEnglish.triggered.connect(self.select_English)

        # When detection method is changed.
        self.tab_signal_type.currentChanged.connect(self.select_detection_method)

        # When parameters are update in fdem interface.
        self.le_detector_radius.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_detector_current.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_detector_frequency.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_detector_pitch.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_detector_roll.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_target_conductivity.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_target_permeability.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_target_radius.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_target_length.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_target_pitch.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_target_roll.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_target_position_x.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_target_position_y.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_target_position_z.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_collection_spacing.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_collection_height.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_collection_SNR.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_collection_x_min.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_collection_x_max.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_collection_y_min.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.le_collection_y_max.editingFinished.connect(self.get_fdem_simulation_parameters)
        self.cb_collection_direction.currentIndexChanged.connect(self.get_fdem_simulation_parameters)

        # When run forward simulation button is clicked in fdem.
        self.pb_run_fdem_forward_simulation.clicked.connect(self.run_fdem_forward_calculate)

        # When run inversion button is clicked in fdem.
        self.pb_run_fdem_inversion.clicked.connect(self.run_fdem_inversion)

    def select_Chinese(self):
        """
        Translate English on the interface into Chinese.

        Referenceres
        ------------
        https://blog.csdn.net/CholenMine/article/details/80725088

        """
        self.trans.load('./translation/zh_CN')
        _app = QApplication.instance()
        _app.installTranslator(self.trans)
        self.retranslateUi(self)

        self.result.current_language = 'cn'

    def select_English(self):
        _app = QApplication.instance()
        _app.removeTranslator(self.trans)
        self.retranslateUi(self)
        self.result.current_language = 'en'

    def select_detection_method(self):
        """
        When detection method is changed, this function will be called.
        """

        if self.tab_signal_type.currentWidget() == self.tab_FDEM:
            self.pb_run_fdem_forward_simulation.setVisible(True)
            self.pb_run_fdem_inversion.setVisible(True)
            self.pb_run_tdem_forward_simulation.setVisible(False)
            self.pb_run_tdem_classification.setVisible(False)

            self.result.current_method = 'fdem'
            self.get_fdem_simulation_parameters()
        else:
            self.pb_run_fdem_forward_simulation.setVisible(False)
            self.pb_run_fdem_inversion.setVisible(False)
            self.pb_run_tdem_forward_simulation.setVisible(True)
            self.pb_run_tdem_classification.setVisible(True)

            self.result.current_method = 'tdem'

    def get_fdem_simulation_parameters(self):
        """
        When parameters are update in fdem interface, the parameters used for
        calculation will be updated.

        """

        self.detector = Detector(
            float(self.le_detector_radius.text()),
            float(self.le_detector_current.text()),
            float(self.le_detector_frequency.text()),
            float(self.le_detector_pitch.text()),
            float(self.le_detector_roll.text())
        )
        self.target = Target(
            float(self.le_target_conductivity.text()),
            float(self.le_target_permeability.text()),
            float(self.le_target_radius.text()),
            float(self.le_target_pitch.text()),
            float(self.le_target_roll.text()),
            float(self.le_target_length.text()),
            float(self.le_target_position_x.text()),
            float(self.le_target_position_y.text()),
            float(self.le_target_position_z.text())
        )
        self.collection = Collection(
            float(self.le_collection_spacing.text()),
            float(self.le_collection_height.text()),
            float(self.le_collection_SNR.text()),
            float(self.le_collection_x_min.text()),
            float(self.le_collection_x_max.text()),
            float(self.le_collection_y_min.text()),
            float(self.le_collection_y_max.text()),
            self.cb_collection_direction.currentText()
        )

        # Update the detection scenario.
        self.result.check_FPara_change = False
        show_fdem_detection_scenario(self.fig_scenario, self.target, self.collection)
        self.canvas_scenario.draw()

    def run_fdem_forward_calculate(self):
        """
        When 'run forward simulation' button is clicked in fdem interface, this
        function will ba called.

        """
        self.get_fdem_simulation_parameters()
        # Disable buttons.
        self.pb_run_fdem_forward_simulation.setEnabled(False)
        self.pbar_rfs.setVisible(True)

        # Update the parameters in the thread.
        save = False
        if self.cb_func_save_data.isChecked():
            save = True

        # Output begin
        text = self.result.output_forward_begin()
        self.tb_output_box.setText(text)

        # show the fdem_forward porgram is running by progressbar
        self.pbar_rfs.setMinimum(0)  # let the progressbar to scroll
        self.pbar_rfs.setMaximum(0)  # let the progressbar to scroll

        # Start the thread.
        #self.thread_cal_fdem = ThreadCalFdem()
        self.thread_cal_fdem.target = self.target
        self.thread_cal_fdem.detector = self.detector
        self.thread_cal_fdem.collection = self.collection
        self.thread_cal_fdem.save = save
        self.thread_cal_fdem.start()

        self.thread_cal_fdem.trigger.connect(self.run_fdem_forward_result_process)

    def run_fdem_forward_result_process(self, forward_result):
        """
        When the forward calculation of FDEM is finished, the result process
        function will be called.

        """
        self.result.forward_result = forward_result
        self.result.check_FPara_change = True
        handler = FDEMHandler(forward_result, None)

        # Plot discetize
        handler.show_discretize(self.fig_discretize)
        self.canvas_discretize.draw()

        # Plot secondary field.
        handler.show_fdem_mag_map(self.fig_magnetic_field)
        self.canvas_magnetic_field.draw()

        # Output finish information.
        text = self.result.output_forward_end()
        self.tb_output_box.setText(text)
        self.tab_show.setCurrentWidget(self.tab_magnetic_field_data)
        self.pb_run_fdem_forward_simulation.setEnabled(True)

        # let the progressBar stop scrolling,it means the  fdem_forward porgram is stoping
        self.pbar_rfs.setMaximum(100)
        self.pbar_rfs.setValue(100)

    def run_fdem_inversion(self):
        """When 'run fdem inversion' button is clicked in fdem interface, this
        function will ba called.
        """
        if not self.result.check_FPara_change:
            text = self.result.output_check_FPara_change()
            self.tb_output_box.setText(text)

        elif self.result.forward_result is None:
            text = self.result.output_check_mag_data()
            self.tb_output_box.setText(text)
        else:
            self.pb_run_fdem_inversion.setEnabled(False)
            self.pbar_rfi.setVisible(True)
            save = False
            if self.cb_func_save_result.isChecked():
                save = True

            # Update the parameters associated with the optimization algorithm.
            method = self.cb_optimization_algorithm.currentText()

            self.thread_inv_fdem.method = self.cb_optimization_algorithm.currentText()
            self.thread_inv_fdem.iterations = float(self.le_optimization_iterations.text())
            self.thread_inv_fdem.tol = float(self.le_optimization_tol.text())
            self.thread_inv_fdem.forward_result = self.result.forward_result
            self.thread_inv_fdem.save = save

            # Output begin
            text = self.result.output_data_process_begin()
            self.tb_output_box.setText(text)

            self.pbar_rfi.setMinimum(0)  # let the progressbar to scroll
            self.pbar_rfi.setMaximum(0)  # let the progressbar to scroll

            # Start the thread.
            self.thread_inv_fdem.start()
            self.thread_inv_fdem.trigger.connect(self.run_fdem_inv_result_process)

    def run_fdem_inv_result_process(self, inv_result):
        self.result.inv_result = inv_result

        text = self.result.output_fdem_result()
        self.tb_output_box.setText(text)

        self.pb_run_fdem_inversion.setEnabled(True)
        self.pbar_rfi.setMaximum(100)
        self.pbar_rfi.setValue(100)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    m = MainWindow()
    m.show()

    sys.exit(app.exec_())
