# -*- coding: utf-8 -*-
"""
The model class, represent the model in FDEM

Class:
- Model: the implement class of the BaseFDEMModel
"""
__all__ = ['Model']
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
from ..utils import RotationMatrix
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG import maps
import SimPEG.electromagnetics.frequency_domain as fdem

from scipy.constants import mu_0

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


class BaseFDEMModel(metaclass=ABCMeta):
    """the abstract class about the model in FDEM

    Attributes
    ----------
    Survey: class
        the Survey in FDEM

    Methods:
    dpred
        Returns the forward simulation data of the FDEM
    """
    @abstractmethod
    def __init__(self, Survey):
        self.survey = Survey

    @abstractmethod
    def dpred(self):
        pass


class Model(BaseFDEMModel):
    def __init__(self, Survey):
        BaseFDEMModel.__init__(self, Survey)

    def dpred(self):
        target = self.survey.source.target
        collection = self.survey.source.collection

        '''Mesh'''
        # Conductivity in S/m (or resistivity in Ohm m)
        background_conductivity = 1e-6
        air_conductivity = 1e-8

        # Permeability in H/m
        background_permeability = mu_0
        air_permeability = mu_0

        dh = 0.1  # base cell width
        dom_width = 20.0  # domain width
        # num. base cells
        nbc = 2 ** int(np.round(np.log(dom_width / dh) / np.log(2.0)))

        # Define the base mesh
        h = [(dh, nbc)]
        mesh = TreeMesh([h, h, h], x0="CCC")

        # Mesh refinement near transmitters and receivers
        mesh = refine_tree_xyz(
            mesh, collection.receiver_location, octree_levels=[2, 4], method="radial",
            finalize=False
        )

        # Refine core mesh region
        xp, yp, zp = np.meshgrid([-1.5, 1.5], [-1.5, 1.5], [-6, -4])
        xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
        mesh = refine_tree_xyz(mesh, xyz, octree_levels=[0, 6], method="box",
                               finalize=False)

        mesh.finalize()

        '''Maps'''

        # Find cells that are active in the forward modeling (cells below surface)
        ind_active = mesh.gridCC[:, 2] < 0

        # Define mapping from model to active cells
        active_sigma_map = maps.InjectActiveCells(mesh, ind_active,
                                                  air_conductivity)
        active_mu_map = maps.InjectActiveCells(mesh, ind_active, air_permeability)

        # Define model. Models in SimPEG are vector arrays
        N = int(ind_active.sum())
        model = np.kron(np.ones((N, 1)), np.c_[background_conductivity,
                                               background_permeability])

        ind_cylinder = self.getIndicesCylinder(
            [target.position[0], target.position[1], target.position[2]],
            target.radius, target.length, [target.pitch, target.roll], mesh.gridCC
        )
        ind_cylinder = ind_cylinder[ind_active]
        model[ind_cylinder, :] = np.c_[target.conductivity, target.permeability]

        # Create model vector and wires
        model = mkvc(model)
        wire_map = maps.Wires(("sigma", N), ("mu", N))

        # Use combo maps to map from model to mesh
        sigma_map = active_sigma_map * wire_map.sigma
        mu_map = active_mu_map * wire_map.mu

        '''Simulation'''
        simulation = fdem.simulation.Simulation3DMagneticFluxDensity(
            mesh, survey=self.survey.survey, sigmaMap=sigma_map, muMap=mu_map, Solver=Solver
        )
        '''Predict'''

        # Compute predicted data for your model.
        dpred = simulation.dpred(model)
        dpred = dpred * 1e9

        # Data are organized by frequency, transmitter location, then by receiver.
        # We had nFreq transmitters and each transmitter had 2 receivers (real and
        # imaginary component). So first we will pick out the real and imaginary
        # data
        bx_real = dpred[0: len(dpred): 6]
        bx_imag = dpred[1: len(dpred): 6]
        bx_total = np.sqrt(np.square(bx_real) + np.square(bx_imag))
        by_real = dpred[2: len(dpred): 6]
        by_imag = dpred[3: len(dpred): 6]
        by_total = np.sqrt(np.square(by_real) + np.square(by_imag))
        bz_real = dpred[4: len(dpred): 6]
        bz_imag = dpred[5: len(dpred): 6]
        bz_total = np.sqrt(np.square(bz_real) + np.square(bz_imag))

        mag_data = np.c_[mkvc(bx_total), mkvc(by_total), mkvc(bz_total)]

        mag_data = self.mag_data_add_noise(mag_data, collection.SNR)
        data = np.c_[collection.receiver_location, mag_data]

        return data, mesh, sigma_map * model

    def mag_data_add_noise(self, mag_data, snr):
        """add the noise for the mag_data

        Parameters
        ----------
        mag_data : TYPE
            DESCRIPTION.
        snr : TYPE
            DESCRIPTION.

        Returns
        -------
        res: ndarry
        """

        mag_data[:, 0] = self.add_wgn(mag_data[:, 0], snr)
        mag_data[:, 1] = self.add_wgn(mag_data[:, 1], snr)
        mag_data[:, 2] = self.add_wgn(mag_data[:, 2], snr)

        return mag_data

    def add_wgn(self, data, snr):
        """add the noise for the data

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        snr : TYPE
            DESCRIPTION.

        Returns
        -------
        res: ndarry
        """

        ps = np.sum(abs(data) ** 2) / len(data)
        pn = ps / (10 ** ((snr / 10.0)))
        noise = np.random.randn(len(data)) * np.sqrt(pn)
        signal_add_noise = data + noise
        return signal_add_noise

    def getIndicesCylinder(self, center, radius, height, oritation, ccMesh):
        """Create the mesh indices of a custom cylinder

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
        d_foot_to_center = (np.linalg.norm((ccMesh - center) * rotated_vector, axis=1)
                            / np.linalg.norm(rotated_vector))

        # Calculate the distances from each mesh to the central axis of the
        # cylinder
        d_meshcc_to_axis = np.sqrt(np.square(ccMesh - center).sum(axis=1)
                                   - np.mat(np.square(d_foot_to_center)).T)
        d_meshcc_to_axis = np.squeeze(np.array(d_meshcc_to_axis))

        ind1 = d_foot_to_center < height / 2
        ind2 = d_meshcc_to_axis < radius
        ind = ind1 & ind2

        return ind




