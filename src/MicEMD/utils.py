# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:20:36 2020

@author: Wang Zhen
Methods:
- mkvc: Creates a vector with the number of dimension specified
- RotationMatrix: create the Euler rotation matrix
"""
__all__ = ['mkvc', 'RotationMatrix']

import numpy as np
from math import sin, cos, pi


def mkvc(x, numDims=1):
    """Creates a vector with the number of dimension specified

    e.g.::

        a = np.array([1, 2, 3])

        mkvc(a, 1).shape
            > (3, )

        mkvc(a, 2).shape
            > (3, 1)

        mkvc(a, 3).shape
            > (3, 1, 1)

    """
    if type(x) == np.matrix:
        x = np.array(x)

    assert isinstance(x, np.ndarray), "Vector must be a numpy array"

    if numDims == 1:
        return x.flatten(order='F')
    elif numDims == 2:
        return x.flatten(order='F')[:, np.newaxis]
    elif numDims == 3:
        return x.flatten(order='F')[:, np.newaxis, np.newaxis]


def RotationMatrix(theta, phi, psi):
    """
    Convert v in the observation frame to v prime in the object frame or rotate the
    vector about the origin.

    Parameters
    ----------
    theta: float
        pitch angle
    phi: float
        roll angle
    psi: float
        yaw angle

    Returns
    -------
    res: ndarry
        the Euler rotation matrix

    Notes
    -----
    This program rotates order (Z ->y->x), inside rotation.
    The outer spin of each particular order is equivalent to the inner spin of its opposite order and vice versa.
    The rotation matrix of the coordinate system about the origin and the rotation matrix of the vector (coordinate)
    about the origin are each other's transpose. The rotation matrix of the world coordinate system to
    the target coordinate system and the rotation matrix of the target coordinate system to the world
    coordinate system are each other's transpose.

    References
    ----------
    [1] Y  Wan,  Wang Z ,  Wang P , et al. A Comparative Study of Inversion Optimization Algorithms for
    Underground Metal Target Detection[J]. IEEE Access, 2020, PP(99):1-1.

    """

    theta, phi, psi = theta * pi / 180, phi * pi / 180, psi * pi / 180

    Rz = np.mat([[cos(psi), sin(psi), 0],
                 [-sin(psi), cos(psi), 0],
                 [0, 0, 1]])

    Ry = np.mat([[cos(theta), 0, -sin(theta)],
                 [0, 1, 0],
                 [sin(theta), 0, cos(theta)]])

    Rx = np.mat([[1, 0, 0],
                 [0, cos(phi), sin(phi)],
                 [0, -sin(phi), cos(phi)]])

    return Rx * Ry * Rz


