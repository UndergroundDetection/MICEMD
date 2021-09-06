# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 10:02:54 2020
@author: Shi haodong

the graphical user interface

Class:
- MainWindow: the implement class of the GUI
"""

import sys

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTranslator

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from mainwindow_ui import Ui_MainWindow

from result import TFResult
import MicEMD.fdem as f
import MicEMD.tdem as t
from utilities.show import show_fdem_detection_scenario
from utilities.threadSet import ThreadCalFdem, ThreadInvFdem, ThreadCalTdem, ThreadClsTdem
from MicEMD.handler import FDEMHandler, TDEMHandler


class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Attributes
    ----------
    thread_cal_fdem: class
        the fdem forward simulation thread class.
    thread_inv_fdem: class
        the fdem inversion thread class.
    thread_cal_tdem: class
        the tdem forward simulation thread class.
    thread_cls_tdem: class
        the tdem classification thread class.

    Methods
    -------
    initialize:
        initialization of the interface
    connect_slots:
        the slots functions
    select_Chinese:
        response to switch to Chinese language
    select_English:
        response to switch to Chinese language
    select_detection_method:
        response to switch the detection method tab
    get_fdem_simulation_parameters:
        get fdem simulation parameter values from the interface
    run_fdem_forward_calculate:
        call the fdem forward simulation interface to simulate
    run_fdem_inversion:
        call the fdem inversion interface
    run_fdem_forward_result_process:
        call the handler to handle the fdem forward results
    run_fdem_inv_result_process:
        call the handler to handle the fdem inversion results

    get_tdem_simulation_parameters:
        get tdem simulation parameter values from the interface
    run_tdem_forward_calculate:
        call the tdem forward simulation interface to simulate
    run_tdem_forward_result_process:
        call the handler to handle the tdem forward results
    run_tdem_classification:
        call the tdem classification interface
    run_tdem_cls_result_process:
        call the handler to handle the tdem classification results

    """

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        # init
        self.initialize()

        # Connect slots function.
        self.get_fdem_simulation_parameters()
        self.get_tdem_simulation_parameters()
        # self.select_detection_method()

        # define slots function to response to the action
        self.connect_slots()

        # Due to the long time of forward simulation and inversion,
        # the interface will appear suspended animation state,
        # so multi-thread is used to complete forward simulation and inversion
        self.thread_cal_fdem = ThreadCalFdem()  # Define the fdem forward simulation thread class.
        self.thread_inv_fdem = ThreadInvFdem()  # Define the fdem inversion thread class.
        self.thread_cal_tdem = ThreadCalTdem()  # Define the tdem forward simulation thread class.
        self.thread_cls_tdem = ThreadClsTdem()  # Define the tdem classification thread class.

    def initialize(self):

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

        self.fig_sample_response = Figure(figsize=(4.21, 3.91))
        self.canvas_sample_response = FigureCanvasQTAgg(self.fig_sample_response)
        self.gl_sample_data_t.addWidget(self.canvas_sample_response)

        self.fig_cls_result = Figure(figsize=(4.21, 3.91))
        self.canvas_cls_result = FigureCanvasQTAgg(self.fig_cls_result)
        self.gl_classification_result_t.addWidget(self.canvas_cls_result)

        # set the QProcessBar('rfs' represent 'run fdem simulation',
        # 'rfi' represent 'run fdem inversion' ) invisible
        self.pbar_rfs.setVisible(False)
        self.pbar_rfi.setVisible(False)

    def connect_slots(self):
        # the response to select language action
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

        self.le_detector_radius_t.editingFinished.connect(self.get_tdem_simulation_parameters)
        self.le_detector_current_t.editingFinished.connect(self.get_tdem_simulation_parameters)
        self.le_detector_pitch_t.editingFinished.connect(self.get_tdem_simulation_parameters)
        self.le_detector_roll_t.editingFinished.connect(self.get_tdem_simulation_parameters)
        self.le_target_a_min_t.editingFinished.connect(self.get_tdem_simulation_parameters)
        self.le_target_a_max_t.editingFinished.connect(self.get_tdem_simulation_parameters)
        self.le_target_b_min_t.editingFinished.connect(self.get_tdem_simulation_parameters)
        self.le_target_b_max_t.editingFinished.connect(self.get_tdem_simulation_parameters)
        self.le_target_a_split_t.editingFinished.connect(self.get_tdem_simulation_parameters)
        self.le_target_b_split_t.editingFinished.connect(self.get_tdem_simulation_parameters)
        self.le_collection_split_t.editingFinished.connect(self.get_tdem_simulation_parameters)
        self.le_target_b_split_t.editingFinished.connect(self.get_tdem_simulation_parameters)

        # When run forward simulation button is clicked in fdem.
        self.pb_run_fdem_forward_simulation.clicked.connect(self.run_fdem_forward_calculate)

        # When run forward simulation button is clicked in tdem.
        self.pb_run_tdem_forward_simulation.clicked.connect(self.run_tdem_forward_calculate)

        # When run inversion button is clicked in fdem.
        self.pb_run_fdem_inversion.clicked.connect(self.run_fdem_inversion)

        # When run classification button is clicked in tdem.
        self.pb_run_tdem_classification.clicked.connect(self.run_tdem_classification)

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
        self.pbar_rfi.setVisible(False)
        self.pbar_rfs.setVisible(False)
        if self.tab_signal_type.currentWidget() == self.tab_FDEM:
            self.tab_show.setCurrentWidget(self.tab_detection_scenario)
            self.pb_run_fdem_forward_simulation.setVisible(True)
            self.pb_run_fdem_inversion.setVisible(True)
            self.pb_run_tdem_forward_simulation.setVisible(False)
            self.pb_run_tdem_classification.setVisible(False)

            self.result.current_method = 'fdem'
            # self.get_fdem_simulation_parameters()
        else:
            self.tab_show.setCurrentWidget(self.tab_sample_data_t)
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

        self.fdetector = f.Detector(
            float(self.le_detector_radius.text()),
            float(self.le_detector_current.text()),
            float(self.le_detector_frequency.text()),
            float(self.le_detector_pitch.text()),
            float(self.le_detector_roll.text())
        )
        self.ftarget = f.Target(
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
        self.fcollection = f.Collection(
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
        show_fdem_detection_scenario(self.fig_scenario, self.ftarget, self.fcollection)
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
        # self.thread_cal_fdem = ThreadCalFdem()
        self.thread_cal_fdem.target = self.ftarget
        self.thread_cal_fdem.detector = self.fdetector
        self.thread_cal_fdem.collection = self.fcollection
        self.thread_cal_fdem.save = save
        self.thread_cal_fdem.start()

        self.thread_cal_fdem.trigger.connect(self.run_fdem_forward_result_process)

    def get_tdem_simulation_parameters(self):
        """
        When parameters are update in tdem interface, the parameters used for
        calculation will be updated.
        """

        self.tdetector = t.Detector(
            float(self.le_detector_radius_t.text()),
            float(self.le_detector_current_t.text()),
            float(self.le_detector_pitch_t.text()),
            float(self.le_detector_roll_t.text())
        )
        self.ttarget = t.Target(
            self.le_target_material_t.text().split(','),
            self.le_target_shape_t.text().split(','),
            np.array([[696.3028547, 875 * 1e-6, 50000000], [99.47183638, 125 * 1e-6, 14619883.04],
                      [1.000022202, 1.256665 * 1e-6, 37667620.91]]),
            float(self.le_target_a_min_t.text()),
            float(self.le_target_a_max_t.text()),
            float(self.le_target_b_min_t.text()),
            float(self.le_target_b_max_t.text()),
            float(self.le_target_a_split_t.text()),
            float(self.le_target_b_split_t.text())
        )
        self.tcollection = t.Collection(
            int(self.le_collection_split_t.text()),
            float(self.le_collection_SNR_t.text()),
        )

        self.result.check_tPara_change = False

    def run_tdem_forward_calculate(self):
        """
        When 'run forward simulation' button is clicked in tdem interface, this
        function will ba called.

        """

        self.get_tdem_simulation_parameters()

        # Disable buttons.
        self.pb_run_tdem_forward_simulation.setEnabled(False)
        self.pbar_rfs.setVisible(True)

        # Update the parameters in the thread.
        save = False
        if self.cb_func_save_data_t.isChecked():
            save = True

        # Output begin
        text = self.result.output_forward_begin()
        self.tb_output_box.setText(text)

        # show the tdem_forward porgram is running by progressbar
        self.pbar_rfs.setMinimum(0)  # let the progressbar to scroll
        self.pbar_rfs.setMaximum(0)  # let the progressbar to scroll

        self.thread_cal_tdem.target = self.ttarget
        self.thread_cal_tdem.detector = self.tdetector
        self.thread_cal_tdem.collection = self.tcollection
        self.thread_cal_tdem.save = save
        self.thread_cal_tdem.start()

        self.thread_cal_tdem.trigger.connect(self.run_tdem_forward_result_process)

    def run_tdem_forward_result_process(self, forward_result):
        """
        When the forward calculation of TDEM is finished, the result process
        function will be called.

        """

        self.result.forward_result_t = forward_result
        self.result.check_TPara_change = True
        handler = TDEMHandler(target=self.ttarget, collection=self.tcollection)

        if self.thread_cal_tdem.save:
            handler.save_fwd_data_default(forward_result[0])
        handler.save_sample_data_default(forward_result[1], self.fig_sample_response, False, self.thread_cal_tdem.save)
        self.canvas_sample_response.draw()
        self.tab_show.setCurrentWidget(self.tab_sample_data_t)

        # Output finish information.
        text = self.result.output_forward_end()
        self.tb_output_box.setText(text)
        # self.tab_show.setCurrentWidget(self.tab_magnetic_field_data)
        self.pb_run_tdem_forward_simulation.setEnabled(True)

        # let the progressBar stop scrolling,it means the  tdem_forward porgram is stoping
        self.pbar_rfs.setMaximum(100)
        self.pbar_rfs.setValue(100)

    def run_fdem_forward_result_process(self, forward_result):
        """
        When the forward calculation of FDEM is finished, the result process
        function will be called.

        """
        self.result.forward_result = forward_result
        self.result.check_FPara_change = True
        handler = FDEMHandler(target=self.ftarget, collection=self.fcollection)
        if self.thread_cal_fdem.save:
            handler.save_fwd_data_default(forward_result[0])

        # Plot discetize
        # handler.show_discretize_default(forward_result[1], forward_result[2], self.fig_discretize, show=False,
        #                                 save=self.thread_cal_fdem.save)
        # self.canvas_discretize.draw()

        # Plot secondary field.
        handler.show_mag_map_default(forward_result[0], self.fig_magnetic_field, show=False,
                                     save=self.thread_cal_fdem.save)
        self.canvas_magnetic_field.draw()

        # Output finish information.
        text = self.result.output_forward_end()
        self.tb_output_box.setText(text)
        self.tab_show.setCurrentWidget(self.tab_magnetic_field_data)
        self.pb_run_fdem_forward_simulation.setEnabled(True)

        # let the progressBar stop scrolling,it means the  fdem_forward porgram is stoping
        self.pbar_rfs.setMaximum(100)
        self.pbar_rfs.setValue(100)

    def run_tdem_classification(self):
        """
        When run fdem inversion' button is clicked in fdem interface, this
        function will ba called.
        """
        if not self.result.check_TPara_change:
            text = self.result.output_check_TPara_change()
            self.tb_output_box.setText(text)

        elif self.result.forward_result_t is None:
            text = self.result.output_check_mag_data()
            self.tb_output_box.setText(text)
        else:
            self.pb_run_tdem_classification.setEnabled(False)
            self.pbar_rfi.setVisible(True)
            save = False
            if self.cb_func_save_result_t.isChecked():
                save = True

            # Update the parameters associated with the optimization algorithm.
            task = self.cb_dr_task_t.currentText()
            dir_method = self.cb_dr_algorithm_t.currentText()
            if dir_method == 'None':
                dir_method = None
            cls_method = self.cb_cls_algorithm_t.currentText()

            self.thread_cls_tdem.dir_method = dir_method
            self.thread_cls_tdem.cls_method = cls_method
            self.thread_cls_tdem.task = task
            self.thread_cls_tdem.forward_result = self.result.forward_result_t
            self.thread_cls_tdem.target = self.ttarget
            self.thread_cls_tdem.collection = self.tcollection
            self.thread_cls_tdem.save = save

            # Output begin
            text = self.result.output_data_process_begin()
            self.tb_output_box.setText(text)

            self.pbar_rfi.setMinimum(0)  # let the progressbar to scroll
            self.pbar_rfi.setMaximum(0)  # let the progressbar to scroll

            # Start the thread.
            self.thread_cls_tdem.start()
            self.thread_cls_tdem.trigger.connect(self.run_tdem_cls_result_process)

    def run_tdem_cls_result_process(self, cls_result):

        self.result.cls_result = cls_result
        handler = TDEMHandler(target=self.ttarget, collection=self.tcollection)
        handler.show_cls_res_default(cls_result, self.thread_cls_tdem.task, self.fig_cls_result, False, self.thread_cls_tdem.save)
        handler.save_cls_res_default(cls_result)
        self.canvas_cls_result.draw()
        self.tab_show.setCurrentWidget(self.tab_classification_result_t)

        text = self.result.output_tdem_result()
        self.tb_output_box.setText(text)

        self.pb_run_tdem_classification.setEnabled(True)
        self.pbar_rfi.setMaximum(100)
        self.pbar_rfi.setValue(100)

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
            self.thread_inv_fdem.target = self.ftarget
            self.thread_inv_fdem.detector = self.fdetector
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
        handler = FDEMHandler(target=self.ftarget, collection=self.fcollection)
        handler.show_inv_res_default(inv_result, self.fig_discretize, show=False,
                                     save=self.thread_cal_fdem.save)
        self.canvas_discretize.draw()

        if self.thread_inv_fdem.save:
            handler.save_inv_res_default(inv_result, self.thread_inv_fdem.method)

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
