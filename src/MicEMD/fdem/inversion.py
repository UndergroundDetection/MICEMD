# -*- coding: utf-8 -*-
"""
The inversion class in FDEM

Class:
- Inversion: the implement class of the BaseFDEMInversion

Methods:
- inverse: the interface of the inverse in FDEM
"""

__all__ = ['Inversion', 'inverse']

from abc import ABCMeta
from abc import abstractmethod
import numpy as np
from scipy.constants import mu_0
from ..optimization import *


class BaseFDEMInversion(metaclass=ABCMeta):
    """the abstract inversion base class

    Parameters
    ----------
    data: tuple
        conclude the observed data and the target class and detection class
    method: str
        the name of optimization
    inv_para: dict
        the parameters setting of the optimization

    Methods
    -------
    true_properties:
        Returns the ture properties
    run:
        Returns the estimate properties
    error:
        Returns the error between true value and estimate value
    inv_objective_function:
        Returns the calculated objective_function value in x position
    """

    @abstractmethod
    def __init__(self, data, method, inv_para):
        pass

    @abstractmethod
    def true_properties(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def error(self):
        pass

    @abstractmethod
    def inv_objective_function(self):
        pass


class Inversion(BaseFDEMInversion):
    """inverse the properties of the target

    based on the optimization algorithms to solve the inversion problem

    Parameters
    ----------
    data: tuple
        the data conclude the observed data, target class and detector class
    method: str
        the name of the optimization
    inv_para: dict
        the parameters setting of the optimization

    Methods
    -------
    true_properties:
        Returns the true properties of the target
    inv_objective_function:
        Returns the calculated objective_function value in x position
    inv_objectfun_gradient:
        Returns the calculated gradient value in x position
    inv_residual_vector:
        Returns the residual vector.
    inv_get_preicted_data:
        Returns Predicted secondary fields calculated by the dipole model at all receiver_loc.
    inv_forward_calculation:
        Returns  predicted secondary fields according the linear magnetic
        dipole model at receiver_loc.
    inv_residual_vector_grad:
        Returns the gradient of all the residual vectors.
    inv_forward_grad:
        Returns the gradient of inv_forward_calculation()
    polar_tensor_to_properties:
        ransform the polar tensor to properties transform M11, M22, M33, M12, M13, M23
        to polarizability and pitch and roll angle and return
    find_xyz_polarizability_index:
        make the order of eigenvalue correspond to the polarizability order
    run:
        run the process of the inversion and return the estimate values
    error:
        Returns the error between true value and estimate value

    """

    def __init__(self, data, method, inv_para):  # , x0, iterations, tol, ForwardResult
        """

        Parameters
        ----------
        data: tuple or list
            conclude receiver_location, magnetic data, target and detector
        method: str
            the optimization method
        inv_para: dict
            the parameters of the inversion,conclude the initial value, iteration and Threshold of loss

        """
        BaseFDEMInversion.__init__(self, data, method, inv_para)
        self.method = method
        self.x0 = inv_para['x0']
        self.iterations = inv_para['iterations']
        self.tol = inv_para['tol']
        self.receiver_locations = data[0][:, 0:3]
        self.mag_data = data[0][:, 3:6]
        self.target = data[1]
        self.detector = data[2]
        self.fun = lambda x: self.inv_objective_function(self.detector, self.receiver_locations, self.mag_data, x)
        self.grad = lambda x: self.inv_objectfun_gradient(self.detector, self.receiver_locations, self.mag_data, x)
        self.jacobian = lambda x: self.inv_residual_vector_grad(self.detector, self.receiver_locations, x)

    @property
    def true_properties(self):
        """return the true properties

        Returns
        -------
        res: ndarry
            return the true properties, conclude position, polarizability, pitch and
            roll angle of the target, amount to 8 inverse parameters
        """
        true_properties = np.array(self.target.position)
        true_polarizability = np.abs(self.target.get_principal_axis_polarizability_complex(self.detector.frequency))
        true_properties = np.append(true_properties, true_polarizability)
        true_properties = np.append(true_properties, self.target.pitch)
        true_properties = np.append(true_properties, self.target.roll)
        return true_properties

    def inv_objective_function(self, detector, receiver_locations, true_mag_data, x):
        """Objective function.

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

        residual = self.inv_residual_vector(detector, receiver_locations, true_mag_data, x)
        objective_fun_value = np.square(residual).sum() / 2.0

        return objective_fun_value

    def inv_objectfun_gradient(self, detector, receiver_locations, true_mag_data, x):
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

        rx = self.inv_residual_vector(detector, receiver_locations, true_mag_data, x)
        jx = self.inv_residual_vector_grad(detector, receiver_locations, x)
        grad = rx.T * jx
        grad = np.array(grad)[0]
        return grad

    def inv_residual_vector(self, detector, receiver_locations, true_mag_data, x):
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

        predicted_mag_data = self.inv_get_preicted_data(detector, receiver_locations, x)
        predicted_mag_data = predicted_mag_data.flatten()
        true_mag_data = true_mag_data.flatten()
        residual = predicted_mag_data - true_mag_data

        residual = np.mat(residual).T
        return residual

    def inv_get_preicted_data(self, detector, receiver_locations, x):
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
            B = self.inv_forward_calculation(detector, receiver_loc, x)
            predicted_mag_data[idx, :] = B.T

        return predicted_mag_data

    def inv_forward_calculation(self, detector, receiver_loc, x):
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
        m_d = np.mat([0, 0, detector.mag_moment]).T
        target_lacation = np.mat(x[0:3]).T
        M11, M22, M33, M12, M13, M23 = x[3:]
        M = np.mat([[M11, M12, M13], [M12, M22, M23], [M13, M23, M33]])

        r_dt = target_lacation - np.mat(receiver_loc).T
        # Calculate primary field using formaula (2)
        H = 1 / (4 * np.pi) * (
                (3 * r_dt * (m_d.T * r_dt)) / (np.linalg.norm(r_dt)) ** 5
                - m_d / (np.linalg.norm(r_dt)) ** 3
        )
        # Calculate induced magnetic moment
        m_t = M * H

        # Calculate secondary field using formula (5)
        r_td = - r_dt
        B = mu_0 / (4 * np.pi) * (
                (3 * r_td * (m_t.T * r_td)) / (np.linalg.norm(r_td)) ** 5
                - m_t / (np.linalg.norm(r_td)) ** 3
        )
        B = abs(B) * 1e9
        # B = np.array(B) * 1e9  # 复数

        return B

    def inv_residual_vector_grad(self, detector, receiver_locations, x):
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

        grad = np.mat(np.zeros((3 * len(receiver_locations), len(x))))
        for i, receiver_loc in enumerate(receiver_locations):
            grad[3 * i:3 * i + 3, :] = self.inv_forward_grad(detector, receiver_loc, x)

        return grad

    def inv_forward_grad(self, detector, receiver_loc, x):
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

        epsilon = 1e-9
        grad = np.mat(np.zeros((3, len(x))))
        ei = [0 for i in range(len(x))]
        ei = np.array(ei)

        for i in range(len(x)):
            ei[i] = 1.0
            d = epsilon * ei
            grad[:, i] = (
                    (self.inv_forward_calculation(detector, receiver_loc, np.array(x) + d)
                     - self.inv_forward_calculation(detector, receiver_loc, x)) / d[i]
            )
            # a = inv_forward_calculation(detector, receiver_loc, np.array(x) + d)
            # b = inv_forward_calculation(detector, receiver_loc, x)
            # grad[:, i] = (a - b) / d[i]
            ei[i] = 0.0

        return grad

    def polar_tensor_to_properties(self, polar_tensor_vector):
        """transform the polar tensor to properties
        transform M11, M22, M33, M12, M13, M23 to polarizability
        and pitch and roll angle

        Parameters
        ----------
        polar_tensor_vector : numpy.array, size=6
            M11, M22, M33, M12, M13, M23.

        Returns
        -------
        None.

        """

        M11, M22, M33, M12, M13, M23 = polar_tensor_vector[:]
        M = np.mat([[M11, M12, M13], [M12, M22, M23], [M13, M23, M33]])
        # print(M)
        eigenvalue, eigenvector = np.linalg.eig(M)
        # because of bx=by, so we can know which is bx,by,bz
        xyz_polar_index = self.find_xyz_polarizability_index(eigenvalue)
        numx = int(xyz_polar_index[0])
        numy = int(xyz_polar_index[1])
        numz = int(xyz_polar_index[2])
        xyz_eigenvalue = np.array([eigenvalue[numx], eigenvalue[numy], eigenvalue[numz]])
        # print(xyz_eigenvalue)
        xyz_eigenvector = np.mat(np.zeros((3, 3)))
        xyz_eigenvector[:, 0] = eigenvector[:, numx]
        xyz_eigenvector[:, 1] = eigenvector[:, numy]
        xyz_eigenvector[:, 2] = eigenvector[:, numz]
        # print(xyz_eigenvector)
        # we suppose the pitch angle and roll angle > 0
        if xyz_eigenvector[0, 2] > 1e-3:
            xyz_eigenvector[:, 2] = - xyz_eigenvector[:, 2]
        pitch = np.arcsin(-xyz_eigenvector[0, 2])

        # roll = np.arcsin(xyz_eigenvector[1, 2] / np.cos(pitch))
        # 假定角度都为正
        tmp = xyz_eigenvector[1, 2] / np.cos(pitch)
        # tmp = -tmp if tmp < 0 else tmp

        roll = np.arcsin(tmp)
        pitch = pitch * 180 / np.pi
        roll = roll * 180 / np.pi

        # # roll = np.arctan(xyz_eigenvector[1, 2] / xyz_eigenvector[2, 2])
        # # 判断是否为0-90度
        #
        # if xyz_eigenvector[0, 2] > 0:
        #     xyz_eigenvector[:, 2] = - xyz_eigenvector[:, 2]
        # if xyz_eigenvector[0, 0] < 0:
        #     xyz_eigenvector[:, 0] = - xyz_eigenvector[:, 0]
        #
        # pitch = np.arcsin(-xyz_eigenvector[0, 2])
        # # pitch = np.arccos(xyz_eigenvector[2, 2]/np.cos(roll))
        # roll = np.arcsin(xyz_eigenvector[1, 2]/np.cos(pitch))
        # pitch = pitch * 180 / np.pi
        # roll = roll * 180 / np.pi
        # print(pitch, roll)

        return np.append(xyz_eigenvalue, [pitch, roll])

    def find_xyz_polarizability_index(self, polarizability):
        """make the order of eigenvalue correspond to the polarizability order

        Parameters
        ----------
        polarizability : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        difference = dict()
        difference['012'] = abs(polarizability[0] - polarizability[1])
        difference['021'] = abs(polarizability[0] - polarizability[2])
        difference['120'] = abs(polarizability[1] - polarizability[2])
        sorted_diff = sorted(difference.items(), key=lambda item: item[1])

        return sorted_diff[0][0]

    def run(self):
        """run the process of the inversion

        Returns
        -------
        res: ndarray
            return the estimate properties[x, y, z, bx, by, bz, pitch, roll]
        """
        estimate_parameters = numopt(self.fun, self.grad, self.jacobian, self.x0, self.iterations, self.method,
                                     self.tol)
        estimate_properties = estimate_parameters[:3]
        est_ploar_and_orientation = self.polar_tensor_to_properties(estimate_parameters[3:])
        estimate_properties = np.append(estimate_properties, est_ploar_and_orientation)
        self.estimate_properties = estimate_properties
        return estimate_properties

    @property
    def error(self):
        """calculate the error of the true value and estimate value

        Returns
        -------
        error: ndarray
            return the error

        """
        return abs(self.true_properties - self.estimate_properties)


def inverse(data, method, inv_para, *args, **kwargs):
    """ the interface is used to handle the process of inversion

    Parameters
    ----------
    data: tuple
        the data conclude the observed data, target class and detector class
    method: str
        the name of the optimization
    inv_para: dict
        the parameters setting of the optimization

    Returns
    -------
    res: dict
        res conclude the predicted value, true value and the error value

    """
    if method in ['Levenberg-Marquardt', 'L-M']:
        method = 'LM'
    if method in ['最速下降', 'Steepest descent']:
        method = 'SD'
    if method in ['共轭梯度', 'Conjugate gradient']:
        method = 'CG'

    if method in ['BFGS', 'CG', 'SD', 'LM']:
        inversion = Inversion(data, method, inv_para)
        res = {}
        estimate_parameters = inversion.run()
        res['pred'] = estimate_parameters
        res['true'] = inversion.true_properties
        res['error'] = inversion.error
        return res
