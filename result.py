# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 10:43:09 2020

@author: Shi Haodong
"""


class Result(object):
    """

    Attributes
    ----------
    current_language : str
        Record the current interface language.
    current_method : str
        Record the current detection method: fdem or tdem.
    check_fdem_mag_data : boolean
        Judge whether the current secondary field data can be directly used in
        inversion. If not, the forward simulation needs to be recalculated
        before the inversion process
    check_tdem_data : boolean
        Judge whether the current secondary field data can be directly used in
        classification.

    fdem_mag_data : numpy.ndarray, shape(N*1)
        Record fdem magnetic field data.
    fdem_receiver_locs : numpy.ndarray, shape(N*3)
        Record fdem detector locations.
    fdem_fdem_optimization_algorithm : str
    fdem_optimization_iterations : int
    fdem_optimization_fval : float
        Record the value at the convergence of the objective function.

    to be continued
    """

    def __init__(self):

        self.current_language = 'en'
        self.current_method = 'fdem'
        self.check_fdem_mag_data = False
        self.check_tdem_mag_data = False

        self.fdem_mag_data = None
        self.fdem_receiver_locs = None
        self.fdem_optimization_algorithm = None
        self.fdem_optimization_iterations = None
        self.fdem_optimization_fval = None
        self.fdem_true_properties = None
        self.fdem_estimate_properties = None
        self.fdem_estimate_error = None

    def output_forward_begin(self):

        if self.current_language == 'en':
            if self.current_method == 'fdem':
                return "FDEM forward simulation is running."
            elif self.current_method == 'tdem':
                return "TDEM forward simulation is running."
        elif self.current_language == 'cn':
            if self.current_method == 'fdem':
                return "频域正向仿真正在运行."
            elif self.current_method == 'tdem':
                return "时域正向仿真正在运行."

    def output_forward_end(self):

        if self.current_language == 'en':
            if self.current_method == 'fdem':
                return "FDEM forward simulation is completed."
            elif self.current_method == 'tdem':
                return "TDEM forward simulation is completed."
        elif self.current_language == 'cn':
            if self.current_method == 'fdem':
                return "频域正向仿真完成."
            elif self.current_method == 'tdem':
                return "时域正向仿真完成."

    def output_data_process_begin(self):

        if self.current_language == 'en':
            if self.current_method == 'fdem':
                return "FDEM inversion is running."
            elif self.current_method == 'tdem':
                return "TDEM classification is running."
        elif self.current_language == 'cn':
            if self.current_method == 'fdem':
                return "频域反演正在运行."
            elif self.current_method == 'tdem':
                return "时域分类正在运行."

    def output_data_process__end(self):

        if self.current_language == 'en':
            if self.current_method == 'fdem':
                return "FDEM inversion is completed."
            elif self.current_method == 'tdem':
                return "TDEM classification is completed."
        elif self.current_language == 'cn':
            if self.current_method == 'fdem':
                return "频域反演完成."
            elif self.current_method == 'tdem':
                return "时域分类完成."

    def output_check_mag_data(self):

        if self.current_language == 'en':
            return ("The forward simulation parameters had been changed. "
                    + "Please run the forward simulation first !")
        elif self.current_language == 'cn':
            return "正向仿真参数已改变, 请先运行正向仿真."

    def output_fdem_result(self):
        """

        """
        if self.current_language == 'en':

            text = "FDEM INVERSION RESULTS\n"
            error = abs(self.fdem_estimate_properties
                        - self.fdem_true_properties)
            text += "Estimate error\n--------------\n"
            text += "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" \
                % (error[0], error[1], error[2], error[3],
                   error[4], error[5], error[6], error[7]
                   )
            text += "Ture properties\n---------------\n"
            text += "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" \
                % (self.fdem_true_properties[0], self.fdem_true_properties[1],
                   self.fdem_true_properties[2], self.fdem_true_properties[3],
                   self.fdem_true_properties[4], self.fdem_true_properties[5],
                   self.fdem_true_properties[6], self.fdem_true_properties[7]
                   )
            text += "Estimate properties\n-------------------\n"
            text += "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" \
                % (self.fdem_estimate_properties[0], self.fdem_estimate_properties[1],
                   self.fdem_estimate_properties[2], self.fdem_estimate_properties[3],
                   self.fdem_estimate_properties[4], self.fdem_estimate_properties[5],
                   self.fdem_estimate_properties[6], self.fdem_estimate_properties[7],
                   )
            return text

        elif self.current_language == 'cn':
            return ""

    def save_mag_data(self):
        pass

    def save_result(self):
        pass
