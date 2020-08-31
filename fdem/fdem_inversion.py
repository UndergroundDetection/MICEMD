# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:17:15 2020

@author: Wang Zhen
"""

import numpy as np
from scipy.constants import mu_0
from scipy.misc import derivative


def inv_forward_calculation(detector, receiver_loc, x):
    """
    Forward calculation in inversion process. It generates predicted secondary
    fields according the linear magnetic dipole model.

    Parameters
    ----------
    detector : class Detector
    receiver_locations : numpy.ndarray, shape(N * 3)
        See fdem_forward_simulation.fdem_forward_simulation receiver_locations.
    x : list, size=9
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

    # Calculate magnetic moment of transmitter coil, target's location,
    # magnetic polarizabilitytensor.
    m_d = np.mat([0, 0, detector.get_mag_moment()]).T
    target_lacation = np.mat(x[0:3]).T
    M11, M12, M13, M22, M23, M33 = x[3:]
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


def inv_get_preicted_data(detector, receiver_locations, x):
    """
    Forward calculation in inversion process. It generates predicted secondary
    fields according the linear magnetic dipole model.

    Parameters
    ----------
    detector : class Detector
    receiver_locations : numpy.ndarray, shape(N * 3)
        See fdem_forward_simulation.fdem_forward_simulation receiver_locations.
    x : list, size=9
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

    for idx, receiver_loc in enumerate(receiver_locations):
        B = inv_forward_calculation(detector, receiver_loc, x)
        predicted_mag_data[idx, :] = B.T

    return predicted_mag_data


def inv_forward_grad(detector, receiver_loc, x):
    """


    Parameters
    ----------
    detector : TYPE
        DESCRIPTION.
    receiver_locations : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

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


def inv_residual_vector(detector, receiver_locations, true_mag_data, x):
    """


    Parameters
    ----------
    detector : TYPE
        DESCRIPTION.
    receiver_locations : TYPE
        DESCRIPTION.
    true_mag_data : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    predicted_mag_data = inv_forward_calculation(detector,
                                                 receiver_locations, x)
    predicted_mag_data = predicted_mag_data.flatten()
    true_mag_data = true_mag_data.flatten()
    residual = predicted_mag_data - true_mag_data

    return residual


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
    x : list, size=9
        See fdem_inversion.inv_forward_calculation x.

    Returns
    -------
    objective_fun_value : float
    """

    residual = inv_residual_vector(
        detector, receiver_locations, true_mag_data, x
    )
    objective_fun_value = (residual**2).sum() / 2.0

    return objective_fun_value


def inv_residual_vector_grad(detector, receiver_locations, x):
    """


    Parameters
    ----------
    detector : TYPE
        DESCRIPTION.
    receiver_locations : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    grad = np.mat(np.zeros((3*len(receiver_locations), len(x))))
    for i, receiver_loc in enumerate(receiver_locations):
        grad[3*i:3*i+3, :] = inv_forward_grad(detector, receiver_loc, x)

    return grad


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

    rx = inv_residual_vector(detector, receiver_locations, true_mag_data, x)
    jx = inv_residual_vector_grad(detector, receiver_locations, x)

    grad = jx.T * rx.T

    return grad


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
