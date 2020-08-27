# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:20:36 2020

@author: Wang Zhen
"""

import numpy as np
from math import sin, cos, pi


def getIndicesCylinder(center, radius, height, oritation, ccMesh):
    """
    Create the mesh indices of a custom cylinder

    Parameters
    ----------
    center : TYPE
        DESCRIPTION.
    radius : TYPE
        DESCRIPTION.
    height : TYPE
        DESCRIPTION.
    oritation : TYPE
        DESCRIPTION.
    ccMesh : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    References
    ----------
    https://zhuanlan.zhihu.com/p/24760577

    """

    # Get the direction vector of the central axis of the cylinder
    init_vector = np.mat([0, 0, 1]).T
    Rotation_mat = RotationMatrix(oritation[0], oritation[1], 0)
    rotated_vector = Rotation_mat * init_vector

    # Define the points
    center = np.mat(center)
    ccMesh = np.mat(ccMesh)

    # Calculate the all distances from the midpoint of the central axis of a
    # cylinder to the perpendicular foot that from each mesh to the central
    # axis of the cylinder
    d_foot_to_center = (np.linalg.norm((ccMesh-center)*rotated_vector, axis=1)
                        / np.linalg.norm(rotated_vector))

    # Calculate the distances from each mesh to the central axis of the
    # cylinder
    d_meshcc_to_axis = np.sqrt(np.square(ccMesh-center).sum(axis=1)
                               - np.mat(np.square(d_foot_to_center)).T)
    d_meshcc_to_axis = np.squeeze(np.array(d_meshcc_to_axis))

    ind1 = d_foot_to_center < height / 2
    ind2 = d_meshcc_to_axis < radius
    ind = ind1 & ind2

    return ind


def RotationMatrix(theta, phi, psi):
    '''将观测坐标系中的向量v转换成物体坐标系中向量v'或者将向量(坐标)绕原点旋转.

    Parameters
    ----------

    Returns
    -------

    Notes
    -----
    此程序旋转顺序(z->y->x),内旋.
    .每种特定顺序的外旋等价于其相反顺序的内旋,反之亦然.
    .坐标系绕原点旋转的旋转矩阵与向量（坐标）绕原点旋转的旋转矩阵互为转置.
    .世界坐标系向目标坐标系旋转的旋转矩阵与目标坐标系向世界坐标系旋转的旋转矩阵互为转置.

    References
    ----------

    Examples
    --------

    '''

    theta, phi, psi = theta*pi/180, phi*pi/180, psi*pi/180

    Rz = np.mat([[cos(psi), sin(psi), 0],
                [-sin(psi), cos(psi), 0],
                [    0    ,    0    , 1]])

    Ry = np.mat([[cos(theta), 0, -sin(theta)],
                 [    0     , 1,      0     ],
                 [sin(theta), 0,  cos(theta)]])

    Rx = np.mat([[1,     0    ,    0    ],
                 [0,  cos(phi), sin(phi)],
                 [0, -sin(phi), cos(phi)]])

    return Rx * Ry * Rz


def mag_data_add_noise(mag_data, snr):
    """


    Parameters
    ----------
    mag_data : TYPE
        DESCRIPTION.
    snr : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    mag_data[:, 0] = add_wgn(mag_data[:, 0], snr)
    mag_data[:, 1] = add_wgn(mag_data[:, 1], snr)
    mag_data[:, 2] = add_wgn(mag_data[:, 2], snr)

    return mag_data


def add_wgn(data, snr):
    """


    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    snr : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    ps = np.sum(abs(data)**2)/len(data)
    pn = ps/(10**((snr/10.0)))
    noise = np.random.randn(len(data)) * np.sqrt(pn)
    signal_add_noise = data + noise
    return signal_add_noise
