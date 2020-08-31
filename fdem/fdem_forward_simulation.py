# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:28:31 2020

@author: Wang Zhen
"""
import numpy as np

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG import maps
import SimPEG.electromagnetics.frequency_domain as fdem

from scipy.constants import mu_0

from utils import getIndicesCylinder

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


class Detector (object):
    """

    Attributes
    ----------

    """

    def __init__(self, radius, current, frequency, pitch, roll):

        self.radius = radius
        self.current = current
        self.frequency = frequency
        self.pitch = pitch
        self.roll = roll

    def get_mag_moment(self):
        """
        Calculate magnetic moment value of transmitter coil according to
        detector parameters.

        Returns
        -------
        TYPE : float
            The value of magnetic moment of z axis.

        """

        return self.current * np.pi * self.radius**2


class Target (object):
    """Refer in particular to a cylindrical target.

    Attributes
    ----------

    """

    def __init__(self, conductivity, permeability, radius, pitch, roll,
                 length, position_x, position_y, position_z):

        self.conductivity = conductivity
        self.permeability = permeability
        self.radius = radius
        self.pitch = pitch
        self.roll = roll
        self.length = length
        self.position = [position_x, position_y, position_z]

    def get_principal_axis_polarizability(self, frequency):
        """
        Calculate the principal axis polarizabilities of the target according
        to the target's parameters and the frequency of primary field.

        Parameters
        ----------
        frequency : float
            Frequency of the primary field.

        Returns
        -------
        polarizability_x : float
            Polarizability of x axis.
        polarizability_y : float
            Polarizability of y axis.
        polarizability_z : float
            Polarizability of z axis.

        """
        pass


class Collection (object):
    """

    Attributes
    ----------

    """

    def __init__(self, spacing, height, SNR, x_min, x_max, y_min, y_max,
                 collection_direction):

        self.spacing = spacing
        self.height = height
        self.SNR = SNR
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.collection_direction = collection_direction


def fdem_forward_simulation(detector, target, collection):
    """
    Get FDEM simulation data using SimPEG.

    Parameters
    ----------
    detector : class Detector
    target : class Target
    collention : class Collection

    Returns
    -------
    receiver_locations ： numpy.ndarray, shape(N * 3)
        All acquisition locations of the detector. Each row represents an
        acquisition location and the three columns represent x, y and z axis
        locations of an acquisition location.
    mag_data : numpy.ndarray, shape(N*3)
        All secondary fields of acquisition locations.
    mesh : SimPEG mesh
    mapped_model : numpy.ndarray

    References
    ----------
    https://docs.simpeg.xyz/content/tutorials/01-models_mapping/plot_1_tensor_models.html#sphx-glr-content-tutorials-01-models-mapping-plot-1-tensor-models-py
    http://discretize.simpeg.xyz/en/master/tutorials/mesh_generation/4_tree_mesh.html#sphx-glr-tutorials-mesh-generation-4-tree-mesh-py
    https://docs.simpeg.xyz/content/tutorials/07-fdem/plot_fwd_2_fem_cyl.html#sphx-glr-content-tutorials-07-fdem-plot-fwd-2-fem-cyl-py
    https://docs.simpeg.xyz/content/examples/05-fdem/plot_inv_fdem_loop_loop_2Dinversion.html#sphx-glr-content-examples-05-fdem-plot-inv-fdem-loop-loop-2dinversion-py

    """

    # Frequencies being predicted
    frequencies = [detector.frequency]

    # Conductivity in S/m (or resistivity in Ohm m)
    background_conductivity = 1e-3
    air_conductivity = 1e-8

    # Permeability in H/m
    background_permeability = mu_0
    air_permeability = mu_0

    """Survey"""

    # Defining transmitter locations
    acquisition_spacing = collection.spacing
    acq_area_xmin, acq_area_xmax = collection.x_min, collection.x_max
    acq_area_ymin, acq_area_ymax = collection.y_min, collection.y_max
    Nx = int((acq_area_xmax-acq_area_xmin)/acquisition_spacing + 1)
    Ny = int((acq_area_ymax-acq_area_ymin)/acquisition_spacing + 1)
    xtx, ytx, ztx = np.meshgrid(np.linspace(acq_area_xmin, acq_area_xmax, Nx),
                                np.linspace(acq_area_ymin, acq_area_ymax, Ny),
                                [collection.height])
    source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
    ntx = np.size(xtx)

    # Define receiver locations
    xrx, yrx, zrx = np.meshgrid(np.linspace(acq_area_xmin, acq_area_xmax, Nx),
                                np.linspace(acq_area_ymin, acq_area_ymax, Ny),
                                [collection.height])
    receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]

    # Create empty list to store sources
    source_list = []

    # Each unique location and frequency defines a new transmitter
    for ii in range(len(frequencies)):
        for jj in range(ntx):

            # Define receivers of different type at each location
            bxr_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
                receiver_locations[jj, :], "x", "real"
            )
            bxi_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
                receiver_locations[jj, :], "x", "imag"
            )
            byr_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
                receiver_locations[jj, :], "y", "real"
            )
            byi_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
                receiver_locations[jj, :], "y", "imag"
            )
            bzr_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
                receiver_locations[jj, :], "z", "real"
            )
            bzi_receiver = fdem.receivers.PointMagneticFluxDensitySecondary(
                receiver_locations[jj, :], "z", "imag"
            )
            receivers_list = [bxr_receiver, bxi_receiver, byr_receiver,
                              byi_receiver, bzr_receiver, bzi_receiver]

            # Must define the transmitter properties and associated receivers
            source_list.append(
                fdem.sources.MagDipole(
                    receivers_list,
                    frequencies[ii],
                    source_locations[jj],
                    orientation="z",
                    moment=detector.get_mag_moment()
                )
            )

    survey = fdem.Survey(source_list)

    '''Mesh'''

    dh = 0.1  # base cell width
    dom_width = 20.0  # domain width
    # num. base cells
    nbc = 2 ** int(np.round(np.log(dom_width / dh) / np.log(2.0)))

    # Define the base mesh
    h = [(dh, nbc)]
    mesh = TreeMesh([h, h, h], x0="CCC")

    # Mesh refinement near transmitters and receivers
    mesh = refine_tree_xyz(
        mesh, receiver_locations, octree_levels=[2, 4], method="radial",
        finalize=False
    )

    # Refine core mesh region
    xp, yp, zp = np.meshgrid([-1.5, 1.5], [-1.5, 1.5], [-4, -2])
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

    ind_cylinder = getIndicesCylinder(
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
        mesh, survey=survey, sigmaMap=sigma_map, muMap=mu_map, Solver=Solver
    )

    '''Predict'''

    # Compute predicted data for a your model.
    dpred = simulation.dpred(model)
    dpred = dpred * 1e9

    # Data are organized by frequency, transmitter location, then by receiver.
    # We had nFreq transmitters and each transmitter had 2 receivers (real and
    # imaginary component). So first we will pick out the real and imaginary
    # data
    bx_real = dpred[0: len(dpred): 6]
    bx_imag = dpred[1: len(dpred): 6]
    bx_total = np.sqrt(np.square(bx_real)+np.square(bx_imag))
    by_real = dpred[2: len(dpred): 6]
    by_imag = dpred[3: len(dpred): 6]
    by_total = np.sqrt(np.square(by_real)+np.square(by_imag))
    bz_real = dpred[4: len(dpred): 6]
    bz_imag = dpred[5: len(dpred): 6]
    bz_total = np.sqrt(np.square(bz_real)+np.square(bz_imag))

    mag_data = np.c_[mkvc(bx_total), mkvc(by_total), mkvc(bz_total)]

    return receiver_locations, mag_data, mesh, sigma_map*model
