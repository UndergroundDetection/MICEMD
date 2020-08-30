# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:17:15 2020

@author: Wang Zhen
"""

import numpy as np


def inv_forward_calculation(detector, receiver_locations, x):
    """
    Forward calculation in inversion process. It generates predicted secondary
    fields according the linear magnetic dipole model.

    Parameters
    ----------
    detector : class Detector
    receiver_locations : numpy.array, size=N
        See fdem_forward_simulation.fdem_forward_simulation receiver_locations.
    x : numpy.matrix, shape(9,1)
        target's parameters in inversion process.
        position x, y, z, polarizability M11, M12, M13, M22, M23, M33.

    Returns
    -------
    predicted_mag_datamag_data : numpy.ndarray, shape(N*3)
        Predicted secondary fields.

    References
    ----------
    Wan Y, Wang Z, Wang P, et al. A Comparative Study of Inversion Optimization
    Algorithms for Underground Metal Target Detection[J]. IEEE Access, 2020, 8:
    126401-126413.
    """

    predicted_mag_data = np.zeros((len(receiver_locations), 3))

    # Calculate magnetic moment of transmitter coil, target's location,
    # magnetic polarizabilitytensor.
    m_d = np.mat([0, 0, detector.get_mag_moment()]).T
    target_lacation = x[0:3, :]
    M11, M12, M13, M22, M23, M33 = (x[3, 0], x[4, 0], x[5, 0], x[6, 0],
                                    x[7, 0], x[8, 0])
    M = np.mat([[M11, M12, M13], [M12, M22, M23], [M13, M23, M33]])

    for receiver_loc in receiver_locations:
        r_dt = 


    return predicted_mag_data


def inv_objective_function(detector, receiver_locations, true_mag_data, x):
    """
    Calculate the residual and the objective function.

    Parameters
    ----------
    detector : class Detector
    receiver_locations : numpy.ndarray, shape(N*3)
        See fdem_forward_simulation.fdem_forward_simulation receiver_locations.
    true_mag_data : numpy.ndarray, shape(N*3)
        See fdem_forward_simulation.fdem_forward_simulation mag_data.
    x : numpy.array, size=9
        See fdem_inversion.inv_forward_calculation x.

    Returns
    -------
    objective_fun_value : float
    """

    objective_fun_value = None

    return objective_fun_value


def inv_objectfun_gradient(detector, receiver_locations, true_mag_data, x):
    """


    Parameters
    ----------
    detector : class Detector
    receiver_locations : numpy.ndarray, shape(N*3)
        See fdem_forward_simulation.fdem_forward_simulation receiver_locations.
    true_mag_data : numpy.ndarray, shape(N*3)
        See fdem_forward_simulation.fdem_forward_simulation mag_data.
    x : numpy.array, size=9
        See fdem_inversion.inv_forward_calculation x.

    Returns
    -------
    gradient : numpy.array, size=9
        The partial derivative of the objective function with respect to nine
        parameters.

    """

    gradient = None

    return gradient


def fdem_inversion(method, iterations, tol):
    """
    Call optimization algorithms.

    Parameters
    ----------
    method : str
        The name of optimization algorithms.
    iterations : int
        Maximum iterations of optimization algorithms.
    tol : float
        Tolerance of optimization algorithms.

    Returns
    -------
    estimate_parameters : numpy.array, size=9
    """

    estimate_parameters = None

    return estimate_parameters
