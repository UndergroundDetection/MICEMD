# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:17:15 2020

@author: Wang Zhen
"""


def inv_forward_calculation(detector, receiver_locations, x):
    """
    Forward calculation in inversion process. It generates predicted secondary
    fields according the linear magnetic dipole model.

    Parameters
    ----------
    detector : class Detector
    receiver_locations : numpy.ndarray
        See fdem_forward_simulation.fdem_forward_simulation receiver_locations.
    x : numpy.array, size=9
        target's parameters in inversion process.
        position x, y, z, polarizability M11, M12, M13, M22, M23, M33.

    Returns
    -------
    predicted_mag_datamag_data : numpy.ndarray, shape(N*3)
        Predicted secondary fields.
    """

    predicted_mag_data = None

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
