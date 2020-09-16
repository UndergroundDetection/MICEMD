# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 10:02:54 2020

@author: Wang Zhen
"""

import sys
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTranslator, QThread, pyqtSignal

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from discretize import TreeMesh

from mainwindow_ui import Ui_MainWindow
from fdem.fdem_forward_simulation import Detector, Target, Collection
from fdem.fdem_forward_simulation import fdem_forward_simulation
from fdem.fdem_inversion import inv_objective_function, inv_objectfun_gradient
from fdem.fdem_inversion import inv_residual_vector_grad, fdem_inversion
from show import show_fdem_detection_scenario
from show import show_fdem_mag_map, show_discretize
from result import Result
from utils import mag_data_add_noise, polar_tensor_to_properties


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):

        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.initialize()
        # Connect slots function.
        self.connect_slots()
        self.get_fdem_simulation_parameters()

    def initialize(self):

        # Set the display position of the mainwindow.
        desktop = QApplication.desktop()
        x = (desktop.width() - self.width()) // 2
        y = (desktop.height()-65 - self.height()) // 2
        self.move(x, y)

        # Desine the translator to translate interface languages.
        self.trans = QTranslator(self)
        # Define the Result class to record the results in the process.
        self.result = Result()
        # Define the fdem forward simulation thread class.
        self.thread_cal_fdem = ThreadCalFdem()
        # Define the fdem inversion thread class.
        self.thread_inv_fdem = ThreadInvFdem()

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

    def connect_slots(self):

        # When the forward calculation of FDEM is finished, the result process
        # function will be called.
        self.thread_cal_fdem.trigger.connect(self.run_fdem_forward_result_process)
        # When the FDEM inversion is finished, the result process function will
        # be called.
        self.thread_inv_fdem.trigger.connect(self.run_fdem_inv_result_process)
        # When language is changed.
        self.actionChinese.triggered.connect(self.select_Chinese)
        self.actionEnglish.triggered.connect(self.select_English)
        # When detection method is changed.
        self.tab_signal_type.currentChanged.connect(
            self.select_detection_method)
        # When run inversion button is clicked in fdem.
        self.pb_run_fdem_inversion.clicked.connect(
            self.run_fdem_inversion)
        # When run forward simulation button is clicked in fdem.
        self.pb_run_fdem_forward_simulation.clicked.connect(
            self.run_fdem_forward_calculate)

        # When parameters are update in fdem interface.
        self.le_detector_radius.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_detector_current.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_detector_frequency.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_detector_pitch.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_detector_roll.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_target_conductivity.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_target_permeability.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_target_radius.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_target_length.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_target_pitch.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_target_roll.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_target_position_x.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_target_position_y.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_target_position_z.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_collection_spacing.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_collection_height.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_collection_SNR.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_collection_x_min.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_collection_x_max.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_collection_y_min.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_collection_y_max.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.cb_collection_direction.currentIndexChanged.connect(
            self.get_fdem_simulation_parameters)
        self.cb_optimization_algorithm.currentIndexChanged.connect(
            self.get_fdem_simulation_parameters)
        self.le_optimization_tol.editingFinished.connect(
            self.get_fdem_simulation_parameters)
        self.le_optimization_iterations.editingFinished.connect(
            self.get_fdem_simulation_parameters)

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

        # Judge whether the current secondary field data can be directly used
        # in classification.
        self.result.check_fdem_mag_data = False

        # Update the detection scenario.
        show_fdem_detection_scenario(self.fig_scenario,
                                     self.target, self.collection)

    def get_tdem_simulation_parameters(self):
        pass

    def run_fdem_forward_calculate(self):
        """
        When 'run forward simulation' button is clicked in fdem interface, this
        function will ba called.

        """

        # Disable buttons.
        self.pb_run_fdem_forward_simulation.setEnabled(False)

        # Update the parameters in the thread.
        self.thread_cal_fdem.detector = self.detector
        self.thread_cal_fdem.target = self.target
        self.thread_cal_fdem.collection = self.collection

        # Output begin
        text = self.result.output_forward_begin()
        self.tb_output_box.setText(text)

        # Start the thread.
        self.thread_cal_fdem.start()

    def run_fdem_forward_result_process(self, receiver_locs,
                                        mag_data, mesh, mapped_model):
        """
        When the forward calculation of FDEM is finished, the result process
        function will be called.

        """

        # Adding noise to the origin magnetic field data.
        mag_data = mag_data_add_noise(mag_data, self.collection.SNR)

        mag_data_total = np.sqrt(np.square(mag_data[:, 0])
                                 + np.square(mag_data[:, 1])
                                 + np.square(mag_data[:, 2]))

        if self.collection.collection_direction in ["x-axis", "x轴"]:
            mag_data_plotting = mag_data[:, 0]
        elif self.collection.collection_direction in ["y-axis", "y轴"]:
            mag_data_plotting = mag_data[:, 1]
        elif self.collection.collection_direction in ["z-axis", "z轴"]:
            mag_data_plotting = mag_data[:, 2]
        elif self.collection.collection_direction in ["Total", "总场"]:
            mag_data_plotting = mag_data_total

        # Plot discetize
        ind = int(mesh.hx.size / 2)
        range_x = [self.collection.x_min, self.collection.x_max]
        range_y = [-6, 0]
        show_discretize(self.fig_discretize, mesh, mapped_model, 'Y', ind,
                        range_x, range_y, self.target.conductivity)
        self.canvas_discretize.draw()

        # Plot secondary field.
        show_fdem_mag_map(self.fig_magnetic_field, receiver_locs,
                          mag_data_plotting)
        self.canvas_magnetic_field.draw()

        # Save the results to result class.
        self.result.fdem_mag_data = mag_data
        self.result.fdem_receiver_locs = receiver_locs

        if self.cb_func_save_data.isChecked():
            self.result.save_mag_data()
        self.result.check_fdem_mag_data = True

        # Output finish information.
        text = self.result.output_forward_end()
        self.tb_output_box.setText(text)
        self.tab_show.setCurrentWidget(self.tab_magnetic_field_data)
        self.pb_run_fdem_forward_simulation.setEnabled(True)

    def run_fdem_classification_calculate(self):
        pass

    def run_fdem_classification_result_process(self):
        pass

    def run_fdem_inversion(self):
        """
        When 'run fdem inversion' button is clicked in fdem interface, this
        function will ba called.
        """

        if self.result.check_fdem_mag_data is False:
            text = self.result.output_check_mag_data()
            self.tb_output_box.setText(text)
        else:
            self.pb_run_fdem_inversion.setEnabled(False)

            # Constructing objective function
            self.thread_inv_fdem.fun = lambda x: inv_objective_function(
                self.detector, self.result.fdem_receiver_locs,
                self.result.fdem_mag_data, x)
            self.thread_inv_fdem.grad = lambda x: inv_objectfun_gradient(
                self.detector, self.result.fdem_receiver_locs,
                self.result.fdem_mag_data, x)
            self.thread_inv_fdem.jacobian = lambda x: inv_residual_vector_grad(
                self.detector, self.result.fdem_receiver_locs, x)

            # Update the parameters associated with the optimization algorithm.
            self.thread_inv_fdem.method = \
                self.cb_optimization_algorithm.currentText()
            self.thread_inv_fdem.iterations = \
                float(self.le_optimization_iterations.text())
            self.thread_inv_fdem.tol = float(self.le_optimization_tol.text())

            # Output begin
            text = self.result.output_data_process_begin()
            self.tb_output_box.setText(text)

            # Start the thread.
            self.thread_inv_fdem.start()

    def run_fdem_inv_result_process(self, estimate_parameters):

        estimate_properties = estimate_parameters[:3]
        est_ploar_and_orientation = \
            polar_tensor_to_properties(estimate_parameters[3:])
        estimate_properties = np.append(estimate_properties,
                                        est_ploar_and_orientation)
        self.result.fdem_estimate_properties = estimate_properties

        true_properties = np.array(self.target.position)
        true_polarizability = self.target.get_principal_axis_polarizability(
            self.detector.frequency)
        true_properties = np.append(true_properties, true_polarizability)
        true_properties = np.append(true_properties, self.target.pitch)
        true_properties = np.append(true_properties, self.target.roll)
        self.result.fdem_true_properties = true_properties

        self.result.fdem_optimization_algorithm = \
            self.cb_optimization_algorithm.currentText()

        text = self.result.output_fdem_result()
        self.tb_output_box.setText(text)
        self.pb_run_fdem_inversion.setEnabled(True)

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


class ThreadCalFdem(QThread):
    """
    fdem forward simulation thread.
    """

    trigger = pyqtSignal(np.ndarray, np.ndarray, TreeMesh, np.ndarray)

    def __init__(self):
        super(ThreadCalFdem, self).__init__()
        self.detector = None
        self.target = None
        self.collection = None

    def run(self):
        receiver_locs, mag_data, mesh, mapped_model = \
            fdem_forward_simulation(
                self.detector, self.target, self.collection)

        self.trigger.emit(receiver_locs, mag_data, mesh, mapped_model)


class ThreadInvFdem(QThread):
    """
    fdem inversion thread.
    """

    trigger = pyqtSignal(np.ndarray)

    def __init__(self):
        super(ThreadInvFdem, self).__init__()
        self.fun = None
        self.grad = None
        self.jacobian = None
        self.method = None
        self.iterations = None
        self.tol = None

    def run(self):
        estimate_parameters = \
            fdem_inversion(self.fun, self.grad, self.jacobian,
                           self.method, self.iterations, self.tol)

        self.trigger.emit(estimate_parameters)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    m = MainWindow()
    m.show()

    sys.exit(app.exec_())
