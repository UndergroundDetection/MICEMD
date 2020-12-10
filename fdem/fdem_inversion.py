# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:17:15 2020

@author: Wang Zhen
"""

import numpy as np
from scipy.constants import mu_0
from scipy.optimize import minimize
from fdem.optimization import Steepest_descent, BFGS, conjugate_gradient, LM


from utils import RotationMatrix


def inv_objective_function(detector, receiver_locations, true_mag_data, x):
    """
    Objective function.

    Parameters
    ----------
    detector : class Detector
    receiver_locations : numpy.ndarray, shape=(N*3), N=len(receiver_locations)
        All acquisition locations of the detector. Each row represents an
        acquisition location and the three columns represent x, y and z axis
        locations of an acquisition location.
    true_mag_data : numpy.ndarray, shape=(N*3)
        All secondary fields of acquisition locations (x, y and z directions).
    x : numpy.array, size=9
        target's parameters in inversion process, including position x, y, z,
        polarizability M11, M22, M33, M12, M13, M23.

    Returns
    -------
    objective_fun_value : float
    """

    residual = inv_residual_vector(
        detector, receiver_locations, true_mag_data, x
    )
    objective_fun_value = np.square(residual).sum() / 2.0

    return objective_fun_value


def inv_objectfun_gradient(detector, receiver_locations, true_mag_data, x):
    """
    The gradient of the objective function with respect to x.

    Parameters
    ----------
    detector : class Detector
    receiver_locations : numpy.ndarray, shape=(N*3)
        See inv_objective_function receiver_locations.
    true_mag_data : numpy.ndarray, shape=(N*3)
        See inv_objective_function true_mag_data.
    x : numpy.array, size=9
        See inv_objective_function x.

    Returns
    -------
    grad : numpy.array, size=9
        The partial derivative of the objective function with respect to nine
        parameters.

    """

    rx = inv_residual_vector(detector, receiver_locations, true_mag_data, x)
    jx = inv_residual_vector_grad(detector, receiver_locations, x)
    grad = rx.T * jx

    grad = np.array(grad)[0]
    return grad


def inv_residual_vector(detector, receiver_locations, true_mag_data, x):
    """
    Construct the residual vector.

    Parameters
    ----------
    detector : class Detector
    receiver_locations : numpy.ndarray, shape=(N*3)
        See inv_objective_function receiver_locations.
    true_mag_data : numpy.ndarray, shape=(N*3)
        See inv_objective_function true_mag_data.
    x : numpy.array, size=9
        See inv_objective_function x.

    Returns
    -------
    residual : numpy.mat, shape=(N*3,1)
        Residual vector.

    """

    predicted_mag_data = inv_get_preicted_data(detector, receiver_locations, x)
    predicted_mag_data = predicted_mag_data.flatten()
    true_mag_data = true_mag_data.flatten()
    residual = predicted_mag_data - true_mag_data

    residual = np.mat(residual).T
    return residual


def inv_get_preicted_data(detector, receiver_locations, x):
    """
    It generates predicted secondary fields at all receiver locations.

    Parameters
    ----------
    detector : class Detector
    receiver_locations : numpy.ndarray, shape=(N*3)
        See inv_objective_function receiver_locations.
    x : numpy.array, size=9
        See inv_objective_function x.

    Returns
    -------
    predicted_mag_data : numpy.ndarray, shape=(N*3)
        Predicted secondary fields.

    """

    predicted_mag_data = np.zeros((len(receiver_locations), 3))

    for idx, receiver_loc in enumerate(receiver_locations):
        B = inv_forward_calculation(detector, receiver_loc, x)
        predicted_mag_data[idx, :] = B.T

    return predicted_mag_data


def inv_forward_calculation(detector, receiver_loc, x):
    """
    Forward calculation in inversion process. It generates predicted secondary
    fields according the linear magnetic dipole model at receiver_loc.

    Parameters
    ----------
    detector : class Detector
    receiver_loc : numpy.array, size=3
        A receiver location.
    x : numpy.array, size=9
        See inv_objective_function x.

    Returns
    -------
    B : numpy.mat, shape=(3*1)
        Predicted secondary field.

    References
    ----------
    Wan Y, Wang Z, Wang P, et al. A Comparative Study of Inversion Optimization
    Algorithms for Underground Metal Target Detection[J]. IEEE Access, 2020, 8:
    126401-126413.
    """

    # Calculate magnetic moment of transmitter coil, target's location,
    # magnetic polarizabilitytensor.
    m_d = np.mat([0, 0, detector.get_mag_moment()]).T
    target_lacation = np.mat(x[0:3]).T
    M11, M22, M33, M12, M13, M23 = x[3:]
    M = np.mat([[M11, M12, M13], [M12, M22, M23], [M13, M23, M33]])

    r_dt = target_lacation - np.mat(receiver_loc).T
    # Calculate primary field using formaula (2)
    H = 1 / (4 * np.pi) * (
        (3 * r_dt * (m_d.T * r_dt)) / (np.linalg.norm(r_dt))**5
        - m_d / (np.linalg.norm(r_dt))**3
    )
    # Calculate induced magnetic moment
    m_t = M * H
    # Calculate secondary field using formula (5)
    r_td = - r_dt
    B = mu_0 / (4 * np.pi) * (
        (3 * r_td * (m_t.T * r_td)) / (np.linalg.norm(r_td))**5
        - m_t / (np.linalg.norm(r_td))**3
    )
    B = abs(B) * 1e9

    return B


def inv_residual_vector_grad(detector, receiver_locations, x):
    """
    Calculate the gradient of all the residual vectors.

    Parameters
    ----------
    detector : class Detector
    receiver_locations : numpy.ndarray, shape=(N * 3)
        See inv_objective_function receiver_locations.
    x : numpy.array, size=9
        See inv_objective_function x.

    Returns
    -------
    grad : numpy.mat, shape=(N*3,9)
        Jacobian matrix of the residual vector.

    """

    grad = np.mat(np.zeros((3*len(receiver_locations), len(x))))
    for i, receiver_loc in enumerate(receiver_locations):
        grad[3*i:3*i+3, :] = inv_forward_grad(detector, receiver_loc, x)

    return grad


def inv_forward_grad(detector, receiver_loc, x):
    """
    Use the difference method to calculate the gradient of
    inv_forward_calculation().

    Parameters
    ----------
    detector : class Detector
    receiver_loc : numpy.array, size=3
        A receiver location.
    x : numpy.array, size=9
        See inv_objective_function x.

    Returns
    -------
    grad : numpy.mat, shape=(3*9)

    """

    epsilon = 1e-8
    grad = np.mat(np.zeros((3, len(x))))
    ei = [0 for i in range(len(x))]
    ei = np.array(ei)

    for i in range(len(x)):
        ei[i] = 1.0
        d = epsilon * ei
        grad[:, i] = (
            (inv_forward_calculation(detector, receiver_loc, np.array(x) + d)
             - inv_forward_calculation(detector, receiver_loc, x)) / d[i]
        )
        # a = inv_forward_calculation(detector, receiver_loc, np.array(x) + d)
        # b = inv_forward_calculation(detector, receiver_loc, x)
        # grad[:, i] = (a - b) / d[i]
        ei[i] = 0.0

    return grad


def fdem_inversion(fun, grad, jacobian, method, iterations, tol):
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
    res.x : numpy.array, size=9
    """

    x0 = np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    if method == "Steepest descent":
        x, fval, grad_val, x_log, y_log, grad_log = Steepest_descent(fun, grad, x0, iterations, tol)
    elif method == "BFGS":
        x, fval, grad_val, x_log, y_log, grad_log = BFGS(fun, grad, x0, iterations, tol)
    elif method == "Conjugate gradient":
        x, fval, grad_val, x_log, y_log, grad_log = conjugate_gradient(fun, grad, x0, iterations, tol)
    else :
        x, fval, grad_val, x_log, y_log, grad_log = LM(fun, grad, jacobian, x0, iterations, tol)

    return x
