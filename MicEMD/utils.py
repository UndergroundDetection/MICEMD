# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:20:36 2020

@author: Wang Zhen
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


# def getIndicesCylinder(center, radius, height, oritation, ccMesh):
#     """
#     Create the mesh indices of a custom cylinder
#
#     Parameters
#     ----------
#     center : TYPE
#         DESCRIPTION.
#     radius : TYPE
#         DESCRIPTION.
#     height : TYPE
#         DESCRIPTION.
#     oritation : TYPE
#         DESCRIPTION.
#     ccMesh : TYPE
#         DESCRIPTION.
#
#     Returns
#     -------
#     None.
#
#     References
#     ----------
#     https://zhuanlan.zhihu.com/p/24760577
#
#     """
#
#     # Get the direction vector of the central axis of the cylinder
#     init_vector = np.mat([0, 0, 1]).T
#     Rotation_mat = RotationMatrix(oritation[0], oritation[1], 0)
#     rotated_vector = Rotation_mat * init_vector
#
#     # Define the points
#     center = np.mat(center)
#     ccMesh = np.mat(ccMesh)
#
#     # Calculate the all distances from the midpoint of the central axis of a
#     # cylinder to the perpendicular foot that from each mesh to the central
#     # axis of the cylinder
#     d_foot_to_center = (np.linalg.norm((ccMesh - center) * rotated_vector, axis=1)
#                         / np.linalg.norm(rotated_vector))
#
#     # Calculate the distances from each mesh to the central axis of the
#     # cylinder
#     d_meshcc_to_axis = np.sqrt(np.square(ccMesh - center).sum(axis=1)
#                                - np.mat(np.square(d_foot_to_center)).T)
#     d_meshcc_to_axis = np.squeeze(np.array(d_meshcc_to_axis))
#
#     ind1 = d_foot_to_center < height / 2
#     ind2 = d_meshcc_to_axis < radius
#     ind = ind1 & ind2
#
#     return ind
#
#
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


# def mag_data_add_noise(mag_data, snr):
#     """
#
#
#     Parameters
#     ----------
#     mag_data : TYPE
#         DESCRIPTION.
#     snr : TYPE
#         DESCRIPTION.
#
#     Returns
#     -------
#     None.
#
#     """
#
#     mag_data[:, 0] = add_wgn(mag_data[:, 0], snr)
#     mag_data[:, 1] = add_wgn(mag_data[:, 1], snr)
#     mag_data[:, 2] = add_wgn(mag_data[:, 2], snr)
#
#     return mag_data
#
#
# def add_wgn(data, snr):
#     """
#
#
#     Parameters
#     ----------
#     data : TYPE
#         DESCRIPTION.
#     snr : TYPE
#         DESCRIPTION.
#
#     Returns
#     -------
#     None.
#
#     """
#
#     ps = np.sum(abs(data) ** 2) / len(data)
#     pn = ps / (10 ** ((snr / 10.0)))
#     noise = np.random.randn(len(data)) * np.sqrt(pn)
#     signal_add_noise = data + noise
#     return signal_add_noise


# def polar_tensor_to_properties(polar_tensor_vector):
#     """
#
#     Parameters
#     ----------
#     polar_tensor_vector : numpy.array, size=6
#         M11, M22, M33, M12, M13, M23.
#
#     Returns
#     -------
#     None.
#
#     """
#
#     M11, M22, M33, M12, M13, M23 = polar_tensor_vector[:]
#     M = np.mat([[M11, M12, M13], [M12, M22, M23], [M13, M23, M33]])
#     eigenvalue, eigenvector = np.linalg.eig(M)
#
#     xyz_polar_index = find_xyz_polarizability_index(eigenvalue)
#     numx = int(xyz_polar_index[0])
#     numy = int(xyz_polar_index[1])
#     numz = int(xyz_polar_index[2])
#     xyz_eigenvalue = np.array([eigenvalue[numx],
#                                eigenvalue[numy], eigenvalue[numz]])
#     xyz_eigenvector = np.mat(np.zeros((3, 3)))
#     xyz_eigenvector[:, 0] = eigenvector[:, numx]
#     xyz_eigenvector[:, 1] = eigenvector[:, numy]
#     xyz_eigenvector[:, 2] = eigenvector[:, numz]
#
#     if xyz_eigenvector[0, 2] > 0:
#         xyz_eigenvector[:, 2] = - xyz_eigenvector[:, 2]
#     pitch = np.arcsin(-xyz_eigenvector[0, 2])
#     roll = np.arcsin(xyz_eigenvector[1, 2] / np.cos(pitch))
#     pitch = pitch * 180 / np.pi
#     roll = roll * 180 / np.pi
#
#     return np.append(xyz_eigenvalue, [pitch, roll])
#
#
# def find_xyz_polarizability_index(polarizability):
#     """
#
#
#     Parameters
#     ----------
#     polarizability : TYPE
#         DESCRIPTION.
#
#     Returns
#     -------
#     None.
#
#     """
#
#     difference = dict()
#     difference['012'] = abs(polarizability[0] - polarizability[1])
#     difference['021'] = abs(polarizability[0] - polarizability[2])
#     difference['120'] = abs(polarizability[1] - polarizability[2])
#     sorted_diff = sorted(difference.items(), key=lambda item: item[1])
#
#     return sorted_diff[0][0]

# print(polar_tensor_to_properties([0.014799,    0.016576,    0.016576,  5.0804e-08,   2.534e-07, -1.0459e-06])),
