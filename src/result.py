# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 10:43:09 2020

@author: Shi Haodong
"""


class TFResult(object):
    """The class of the output in mainwindow.
    It mainly processes the different output in different language.

    Attributes
    ----------
    current_language : str
        Record the current interface language.
    current_method : str
        Record the current detection method: fdem or tdem.
    forward_result: class
        the result of simulation
    inv_result: class
        the result of inversion
    check_FPara_change: bool
        Whether the parameters of fdem forward simulation is changed
    check_TPara_change: bool
        Whether the parameters of  tdem forward simulation is changed

    """

    def __init__(self):

        self.current_language = 'en'
        self.current_method = 'fdem'
        self.forward_result = None
        self.forward_result_t = None
        self.inv_result = None
        self.cls_result = None
        self.check_FPara_change = False
        self.check_TPara_change = False

    def output_forward_begin(self):
        """When the forward begin,the response of the mainwindow

        Returns
        -------
        None

        """
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
        """When the forward end,the response of the mainwindow"""

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
        """indicate the current running status is running"""

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
        """indicate the current running status is completed"""
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
        """Check the current status and print"""
        if self.current_language == 'en':
            return ("The forward simulation is not completed. "
                    + "Please run the forward simulation first !")
        elif self.current_language == 'cn':
            return "正向仿真未完成, 请先运行正向仿真."

    def output_check_FPara_change(self):
        """check whether the parameters of fdem forward simulation is changed"""
        if self.current_language == 'en':
            return ("The forward simulation parameters had been changed. "
                    + "Please run the forward simulation first !")
        elif self.current_language == 'cn':
            return "正向仿真参数已改变, 请先运行正向仿真."

    def output_check_TPara_change(self):
        """check Whether the parameters of tdem forward simulation is changed"""
        if self.current_language == 'en':
            return ("The forward simulation parameters had been changed. "
                    + "Please run the forward simulation first !")
        elif self.current_language == 'cn':
            return "正向仿真参数已改变, 请先运行正向仿真."

    def output_tdem_result(self):
        """print the tdem classification results"""
        text = ''
        accuracy = self.cls_result['accuracy']
        if self.current_language == 'en':
            text = "TDEM CLASSIFICATION RESULTS\n"
            text += "accuracy\n--------------\n"
            text += "%.4f \n" % accuracy
        elif self.current_language == 'cn':
            text = "TDEM的分类结果\n"
            text += "准确度\n--------------\n"
            text += "%.4f \n" % accuracy
        return text

    def output_fdem_result(self):
        """print the fdem inversion results"""
        fdem_estimate_properties = self.inv_result['pred']
        fdem_true_properties = self.inv_result['true']
        if self.current_language == 'en':

            text = "FDEM INVERSION RESULTS\n"
            error = abs(fdem_estimate_properties - fdem_true_properties)
            text += "Estimate error\n--------------\n"
            text += "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" \
                    % (error[0], error[1], error[2], error[3],
                       error[4], error[5], error[6], error[7]
                       )
            text += "Ture properties\n---------------\n"
            text += "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" \
                    % (fdem_true_properties[0], fdem_true_properties[1],
                       fdem_true_properties[2], fdem_true_properties[3],
                       fdem_true_properties[4], fdem_true_properties[5],
                       fdem_true_properties[6], fdem_true_properties[7]
                       )
            text += "Estimate properties\n-------------------\n"
            text += "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" \
                    % (fdem_estimate_properties[0], fdem_estimate_properties[1],
                       fdem_estimate_properties[2], fdem_estimate_properties[3],
                       fdem_estimate_properties[4], fdem_estimate_properties[5],
                       fdem_estimate_properties[6], fdem_estimate_properties[7],
                       )
            return text

        elif self.current_language == 'cn':
            fdem_estimate_properties = self.inv_result.estimate_parameters
            fdem_true_properties = self.inv_result.true_parameters

            text = "FDEM 反演 结果\n"
            error = abs(fdem_estimate_properties - fdem_true_properties)
            text += "估计误差\n--------------\n"
            text += "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" \
                    % (error[0], error[1], error[2], error[3],
                       error[4], error[5], error[6], error[7]
                       )
            text += "真实值\n---------------\n"
            text += "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" \
                    % (fdem_true_properties[0], fdem_true_properties[1],
                       fdem_true_properties[2], fdem_true_properties[3],
                       fdem_true_properties[4], fdem_true_properties[5],
                       fdem_true_properties[6], fdem_true_properties[7]
                       )
            text += "估计值\n-------------------\n"
            text += "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" \
                    % (fdem_estimate_properties[0], fdem_estimate_properties[1],
                       fdem_estimate_properties[2], fdem_estimate_properties[3],
                       fdem_estimate_properties[4], fdem_estimate_properties[5],
                       fdem_estimate_properties[6], fdem_estimate_properties[7],
                       )
            return text


