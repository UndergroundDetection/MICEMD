# -*- coding: utf-8 -*-
"""
The model class, represent the model in FDEM

Class:
- Model: the implement class of the BaseFDEMModel
"""
__all__ = ['Model', 'DipoleModle']

from abc import ABCMeta
from abc import abstractmethod
import numpy as np
from MicEMD.utils import RotationMatrix
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
    ---------
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
    """the model class
    we simulate the FDEM response based on Simpeg in MicEMD

    Parameters
    ----------
    Survey: class
        the Survey class

    Methods
    -------
    dpred:
        Returns the observed data
    mag_data_add_noise:
        add the noise for the mag_data and return
    add_wgn:
        add the noise for the data
    """

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
        if collection.SNR is not None:
            mag_data = self.mag_data_add_noise(mag_data, collection.SNR)
        data = np.c_[collection.receiver_location, mag_data]
        # data = (data, )

        return data

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


class DipoleModle(BaseFDEMModel):
    def __init__(self, Survey):
        BaseFDEMModel.__init__(self, Survey)

    def dpred(self):
        target = self.survey.source.target
        collection = self.survey.source.collection
        detector = self.survey.source.detector
        Pm = np.array([0, 0, detector.mag_moment])[np.newaxis, :]  # (1, 3)
        t_postion = np.array(target.position)
        d_postion = collection.receiver_location
        rdt = np.tile(t_postion, (d_postion.shape[0], 1)) - d_postion  # (m, 3)
        # print(np.linalg.norm(rdt, axis=1)[:, np.newaxis])
        r = np.sqrt(np.sum(rdt ** 2, axis=1))[:, np.newaxis]  # (m, 1)
        Hxyz = 1 / (4 * np.pi) * (
                (3 * np.inner(rdt, Pm) * rdt) / r ** 5 - Pm / r ** 3)  # np.inner(rdt, Pm) (m,1)  (m, 3)

        beta = target.get_principal_axis_polarizability_complex(detector.frequency)  # (3,)
        rotaion = RotationMatrix(target.pitch, target.roll, 0)  # (3, 3)
        m = rotaion @ np.diag(beta) @ rotaion.T  # (3, 3)
        mt = Hxyz @ m  # m,3


        rtd = d_postion - np.tile(t_postion, (d_postion.shape[0], 1))  # (m, 3)
        Bxyz = mu_0 / (4 * np.pi) * (
                (3 * np.sum(np.multiply(rtd, mt), axis=1).A * rtd) / r ** 5 - mt / r ** 3) * 1e9  # m,3 rtd和mt是内积

        mag_data = np.abs(Bxyz).A
        # print(mag_data.shape)
        # mag_data = np.array(Bxyz)
        # print(mag_data.shape)
        if collection.SNR is not None:
            mag_data = self.mag_data_add_noise(mag_data, collection.SNR)
        data = np.c_[collection.receiver_location, mag_data]

        return data

        # target = self.survey.source.target
        # collection = self.survey.source.collection
        # detector = self.survey.source.detector
        # Pm = np.array([0, 0, detector.mag_moment])[np.newaxis, :]  # (1, 3)
        # t_postion = np.mat(target.position).T
        # d_postion = collection.receiver_location
        # m_d = np.mat([0, 0, detector.mag_moment]).T
        #
        #
        # bxyz = []
        #
        # for i in range(d_postion.shape[0]):
        #     r_dt = t_postion - np.mat(d_postion[i]).T
        #     # Calculate primary field using formaula (2)
        #     H = 1 / (4 * np.pi) * (
        #             (3 * r_dt * (m_d.T * r_dt)) / (np.linalg.norm(r_dt)) ** 5
        #             - m_d / (np.linalg.norm(r_dt)) ** 3
        #     )  # 3,1
        #
        #     beta = target.get_principal_axis_polarizability_complex(detector.frequency)  # (3,)
        #     rotaion = RotationMatrix(target.pitch, target.roll, 0)  # (3, 3)
        #     m = rotaion @ np.diag(beta) @ rotaion.T  # (3, 3)
        #     m_t = m * H  # m,3
        #
        #     # Calculate secondary field using formula (5)
        #     r_td = - r_dt
        #     B = mu_0 / (4 * np.pi) * (
        #             (3 * r_td * (m_t.T * r_td)) / (np.linalg.norm(r_td)) ** 5
        #             - m_t / (np.linalg.norm(r_td)) ** 3
        #     )
        #     B = B * 1e9
        #
        #     bxyz.append(B)
        # print(bxyz[0], bxyz[1])

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


if __name__ == '__main__':
    import MicEMD.fdem as fdem

    target = fdem.Target(conductivity=5.71e7, permeability=1.26e-6, radius=0.1, pitch=0,
                         roll=0, length=0.8, position_x=0, position_y=0, position_z=-3)
    detector = fdem.Detector(radius=0.4, current=20, frequency=1000, pitch=0, roll=0)
    collection = fdem.Collection(spacing=0.1, height=0, SNR=20, x_min=-0.1, x_max=0,
                                 y_min=-0.1, y_max=0, collection_direction='z-axis')
    Pm = np.array([0, 0, detector.mag_moment])[np.newaxis, :]  # (1, 3)
    t_postion = np.array(target.position)
    # d_postion = collection.receiver_location
    rdt = t_postion - np.array([-2, -2, 0])
    print(rdt)
    # print(np.linalg.norm(rdt, axis=1)[:, np.newaxis])
    r = np.sqrt(np.sum(rdt ** 2))
    Hxyz = 1 / (4 * np.pi) * (
            (3 * np.inner(rdt, Pm) * rdt) / r ** 5 - Pm / r ** 3)  # np.inner(rdt, Pm) (m,1)  (m, 3)

    beta = target.get_principal_axis_polarizability_complex(detector.frequency)  # (3,)

    rotaion = RotationMatrix(target.pitch, target.roll, 0)  # (3, 3)

    m = rotaion @ np.diag(beta) @ rotaion.T  # (3, 3)
    print('m', m)

    mt = Hxyz @ m  # m,3
    print('mt', mt)

    rtd = np.array([-2, -2, 3])  # (m, 3)
    Bxyz = mu_0 / (4 * np.pi) * (
            (3 * np.sum(np.multiply(rtd, mt), axis=1).A * rtd) / r ** 5 - mt / r ** 3) * 1e9  # m,3 rtd和mt是内积

    mag_data = Bxyz
    print('mag', mag_data)









    # import MicEMD.fdem as fdem
    #
    # target = fdem.Target(conductivity=5.71e7, permeability=1.26e-6, radius=0.1, pitch=0,
    #                      roll=0, length=0.8, position_x=0, position_y=0, position_z=-3)
    # detector = fdem.Detector(radius=0.4, current=20, frequency=1000, pitch=0, roll=0)
    # collection = fdem.Collection(spacing=0.1, height=0, SNR=20, x_min=-0.1, x_max=0,
    #                              y_min=-0.1, y_max=0, collection_direction='z-axis')
    # source = fdem.Source(target, detector, collection)
    # survey = fdem.Survey(source)
    # _model = fdem.DipoleModle(survey)
    # simulation = fdem.Simulation(_model)
    # result = simulation.pred()
    # print(np.abs(result[2]))
    #
    # M11, M22, M33, M12, M13, M23 = np.abs(result[1])
    # M = np.mat([[M11, M12, M13], [M12, M22, M23], [M13, M23, M33]])
    # eigenvalue, eigenvector = np.linalg.eig(M)
    #
    #
    # # because of bx=by, so we can know which is bx,by,bz
    # def find_xyz_polarizability_index(polarizability):
    #     """make the order of eigenvalue correspond to the polarizability order
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
    #
    #
    # xyz_polar_index = find_xyz_polarizability_index(eigenvalue)
    # numx = int(xyz_polar_index[0])
    # numy = int(xyz_polar_index[1])
    # numz = int(xyz_polar_index[2])
    # xyz_eigenvalue = np.array([eigenvalue[numx], eigenvalue[numy], eigenvalue[numz]])
    # xyz_eigenvector = np.mat(np.zeros((3, 3)))
    # xyz_eigenvector[:, 0] = eigenvector[:, numx]
    # xyz_eigenvector[:, 1] = eigenvector[:, numy]
    # xyz_eigenvector[:, 2] = eigenvector[:, numz]
    #
    # if xyz_eigenvector[0, 2] > 0:
    #     xyz_eigenvector[:, 2] = - xyz_eigenvector[:, 2]
    # pitch = np.arcsin(-xyz_eigenvector[0, 2])
    # roll = np.arcsin(xyz_eigenvector[1, 2] / np.cos(pitch))
    # pitch = pitch * 180 / np.pi
    # roll = roll * 180 / np.pi
    #
    # print(np.append(xyz_eigenvalue, [pitch, roll]))
    #
    # a = 97 * 3.5 + 92 * 1.5 + 99 * 1.5 + 72 * 1.0 + 94 * 0.3 + 89 * 3.0 + 92 * 2.0 + 90 * 1.0 + 86 * 3.0 + 91 * 2.0 + 85 * 2.0 + 96 * 4.0 + 94 * 2 + 74 * 1 + 71 * 2 + \
    #       90 * 1 + 81 * 1.5 + 89 * 3.5 + 93 * 0.3 + 85 * 1 + 77 * 3.5 + 89 * 3 + 76 * 4 + 100 * 0.3 \
    #       + 78 * 2 + 87 * 5.5 + 99 * 1 + 78 * 3 + 95 * 2.5 + 84 * 1 + 91 * 1 + 83 * 2 + 85 * 6 + 88 * 2 + 89 * 3.5 + 86 * 1 + 95 * 0.3 + 87 * 3 + 89 * 2 + \
    #       97 * 3 + 97 * 3 + 88 * 3 + 92 * 0.3 + 89 * 1 + 95 * 3 + 92 * 3.5 + 90 * 2 + 95 * 2 + 86 * 3 + 91 * 1 + 85 * 3 + 91 * 3 + 86 * 3 + 80 * 2 + 87 * 2
    #
    # score = 3.5 + 1.5 + 1.5 + 1.0 + 0.3 + 3.0 + 2.0 + 1.0 + 3.0 + 2.0 + 2.0 + 4.0 + 2.0 + 1.0 + 2.0 + 1.0 + 1.5 + 3.5 + \
    #         0.3 + 1.0 + 3.5 + 3.0 + 4.0 + 0.3 + 2.0 + 5.5 + 1.0 + 3.0 + 2.5 + 1.0 + 1.0 + 2.0 + 6.0 + 2.0 + 3.5 + 1.0 + \
    #         0.3 + 3.0 + 2.0 + 3.0 + 3.0 + 3.0 + 0.3 + 1.0 + 3.0 + 3.5 + 2.0 + 2.0 + 3.0 + 1.0 + 3.0 + 3.0 + 3.0 + 2.0 + 2.0
    # print(a*4/(score*100))
    #
    #
    # b = [90, 91, 92, 87, 88, 91, 81, 86, 97, 87, 95, 84, 91]
    # score2 = [2, 3, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    # x = sum(list(map(lambda x,y: x*y, b, score2)))
    #
    # print(x * 4 / (sum(score2)*100))
    #
    # pass
