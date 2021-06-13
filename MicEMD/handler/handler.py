__all__ = ['FDEMHandler', 'TDEMHandler']

from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import pandas as pd
import os
import itertools

from sklearn.metrics import confusion_matrix
from ..utils import RotationMatrix
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from SimPEG.utils import plot2Ddata
import matplotlib as mpl
import matplotlib.pyplot as plt


class FDEMBaseHandler(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    @abstractmethod
    def save_forward(self):
        pass

    @abstractmethod
    def save_inv(self):
        pass


class FDEMHandler(FDEMBaseHandler):
    def __init__(self, **kwargs):
        FDEMBaseHandler.__init__(self)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def save_forward(self):
        pass

    def save_inv(self):
        pass

    def get_save_fdem_dir(self):
        """
        to get the parameters about this target scene to make file save the data about fdem_forward_simulation

        Returns
        -------
        file_name : str
            the file name to save the data about fdem_forward_simulation and fdem_inversion

        """

        collection = self.collection
        target = self.target
        file_name = "T.pos=[{:g},{:g},{:g}];T.R={:g};T.L={:g};T.pitch={:g};T.roll={:g};C.snr={:g};C.sp={:g};C.h={:g};" \
                    "C.x=[{:g},{:g}];" \
                    "C.y=[{:g},{:g}]".format(target.position[0], target.position[1],
                                             target.position[2],
                                             target.radius, target.length, target.pitch,
                                             target.roll,
                                             collection.SNR, collection.spacing, collection.height,
                                             collection.x_min, collection.x_max, collection.y_min,
                                             collection.y_max)
        return file_name

    def save_fwd_data(self, data, file_name=None):
        """save the data of the forward_res
        the function to save the forward res

        Parameters
        ----------
        data: ndarray
            conclude the receiver location and magnetic data
        file_name: str
            the name of the file that you want to save, the Default relative
            path is '../results/fdemResults/forward_res',if you just input
            the file name, it will be saved in the path
        """
        mag_data = pd.DataFrame(data, columns=['x', 'y', 'z', 'hx', 'hy', 'hz'])

        path = os.path.dirname(file_name)
        name = os.path.basename(file_name)

        if path is '':
            path = '../results/fdemResults/forward_res'
        if os.path.exists(path):
            mag_data.to_csv('{}/{}'.format(path, name))
        else:
            os.makedirs(path)
            mag_data.to_csv('{}/{}'.format(path, name))

    def save_inv_res(self, inv_res, file_name):
        """save the inversion result

        Parameters
        ----------
        inv_res: dict
            the res of the inversion, conclude the error, the true value and
            the pred value
        file_name: str
            the name of the file that you want to save, the Default relative
            path is '../results/fdemResults/inverse_res',if you just input
            the file name, it will be saved in the path

        """
        pred = np.array(inv_res['pred'])
        true = np.array(inv_res['true'])
        error = np.array(inv_res['error'])
        property = np.vstack(([true, pred, error]))
        invResult_index = ['True_value', 'Estimate_value', 'Error']
        res = pd.DataFrame(property, columns=['x', 'y', 'z', 'polarizability_1', 'polarizability_2',
                                              'polarizability_3', 'pitch', 'roll'], index=invResult_index)
        if file_name is not None:
            path = os.path.dirname(file_name)
            name = os.path.basename(file_name)
            if path is '':
                path = '../results/fdemResults/inverse_res'
        else:
            path = '../results/fdemResults/inverse_res'
            name = 'inv_res.csv'

        if os.path.exists(path):
            res.to_csv('{}/{}'.format(path, name))
        else:
            os.makedirs(path)
            res.to_csv('{}/{}'.format(path, name))

    def save_fwd_data_default(self, loc_mag):
        """
        the mag_data(contained the detection position and secondary field data)
        saved by '.xls' file.

        Parameters
        ----------
        file_name : str
            the specific path of the fdem_results.
            the path named by the parameters of the detection scene
        Returns
        -------
        None.

        """
        file_name = self.get_save_fdem_dir()

        mag_data = pd.DataFrame(loc_mag, columns=['x', 'y', 'z', 'hx', 'hy', 'hz'])

        path = './results/fdemResults/{}'.format(file_name)
        name = 'mag_data.csv'
        if os.path.exists(path):
            mag_data.to_csv('{}/{}'.format(path, name))
        else:
            os.makedirs(path)
            mag_data.to_csv('{}/{}'.format(path, name))

    def save_inv_res_default(self, inv_res, method):
        """
        the inv_result(contained the true properties,estimate properties and
        errors between them) saved by '.xls' file named by the optimization
        algorithm name + '_invResult'

        Parameters
        ----------
        file_name : str
            the specific path of the fdem_results.
            the path named by the parameters of the detection scene

        Returns
        -------
        None.

        """
        pred = np.array(inv_res['pred'])
        true = np.array(inv_res['true'])
        error = np.array(inv_res['error'])
        property = np.vstack(([true, pred, error]))
        invResult_index = ['True_value', 'Estimate_value', 'Error']
        res = pd.DataFrame(property, columns=['x', 'y', 'z', 'polarizability_1', 'polarizability_2',
                                              'polarizability_3', 'pitch', 'roll'], index=invResult_index)
        file_name = self.get_save_fdem_dir()

        path = './results/fdemResults/{}'.format(file_name)

        inv_result = pd.DataFrame(res,
                                  columns=['x', 'y', 'z', 'polarizability_1', 'polarizability_2', 'polarizability_3',
                                           'pitch', 'roll'],
                                  index=invResult_index)
        inv_filename = method + '_invResult'

        if os.path.exists(path):
            inv_result.to_csv('{}/{}.csv'.format(path, inv_filename))
        else:
            os.makedirs(path)
            inv_result.to_csv('{}/{}.csv'.format(path, inv_filename))

    def show_detection_scenario(self, Target, Collection, show=False, save=False, file_name=None, fig=None):
        """
        in 3D ax show the detection scene,the main part is to show the
        metal cylinder with different posture,the posture of the metal
        cylinder can be showed by the rotation matrix

        Parameters
        ----------
        target : class Target
            Contains the parameters of the target.
        collection : class Collection
            Contains the parameters of the collection process.
        show: bool
            whether to show the fig
        save: bool
            whether to save the fig
        file_name: str
            if save is true, the file path to save, it is '../results/fdemResults/forward_res' defaultly
        fig : matplotlib.figure.Figure
            it is None defaultly

        """

        # show the metal cylinder
        target = Target
        collection = Collection

        u = np.linspace(0, 2 * np.pi, 50)  # Divide the circle into 20 equal parts
        h = np.linspace(-0.5, 0.5, 2)  # Divide the height(1m)  into 2 equal parts,corresponding to the bottom and top
        x = target.radius * np.sin(u)
        y = target.radius * np.cos(u)

        x = np.outer(x, np.ones(len(h)))  # 20*2
        y = np.outer(y, np.ones(len(h)))  # 20*2
        z = np.outer(np.ones(len(u)), h)  # 20*2
        z = z * target.length

        x_rotation = np.ones(x.shape)  # 旋转后的坐标 20*2
        y_rotation = np.ones(y.shape)
        z_rotation = np.ones(z.shape)

        th1 = target.pitch
        th2 = target.roll
        a = np.array(RotationMatrix(th1, th2, 0))  # 3*3 pitch,roll
        for i in range(2):
            r = np.c_[x[:, i], y[:, i], z[:, i]]  # 20*3
            rT = r @ a  # 20*3
            x_rotation[:, i] = rT[:, 0]
            y_rotation[:, i] = rT[:, 1]
            z_rotation[:, i] = rT[:, 2]
        if fig is not None:
            fig.clf()  # Clear the figure in different detection scene
        else:
            fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.view_init(30, 45)

        ax.plot_surface(x_rotation + target.position[0], y_rotation + target.position[1],
                        z_rotation + target.position[2],
                        color='#E7C261', alpha=1, antialiased=False)
        verts = [list(zip(x_rotation[:, 0] + target.position[0], y_rotation[:, 0] + target.position[1],
                          z_rotation[:, 0] + target.position[2]))]
        ax.add_collection3d(Poly3DCollection(verts, facecolors='#E7C261'))
        verts = [list(zip(x_rotation[:, 1] + target.position[0], y_rotation[:, 1] + target.position[1],
                          z_rotation[:, 1] + target.position[2]))]
        ax.add_collection3d(Poly3DCollection(verts, facecolors='#E7C261'))

        scan_x = np.arange(collection.x_min, collection.x_max + 1e-8, collection.spacing)
        scan_y = np.arange(collection.y_min, collection.y_max + 1e-8, collection.spacing)
        scan_x, scan_y = np.meshgrid(scan_x, scan_y)
        scan_z = np.ones(scan_x.shape) * collection.height

        for i in range(scan_x.shape[1]):
            ax.plot(scan_x[:, i], scan_y[:, i], scan_z[:, i], 'black')
            ax.scatter(scan_x[:, i], scan_y[:, i], scan_z[:, i], marker='o', color='w', edgecolors='blue')
        ax.set_xticks(np.arange(collection.x_min, collection.x_max + 1, 1))
        ax.set_yticks(np.arange(collection.y_min, collection.y_max + 1, 1))
        ax.set_xlabel('X/m')
        ax.set_ylabel('Y/m')
        ax.set_zlabel('Z/m')
        ax.set_xlim(collection.x_min, collection.x_max)
        ax.set_ylim(collection.y_min, collection.y_max)
        ax.set_zlim(target.position[2] - 2, collection.height)

        ax.grid(None)  # delete the background grid

        if show:
            plt.show()
        if save:
            if file_name is not None:
                path = os.path.dirname(file_name)
                name = os.path.basename(file_name)
                if path is '':
                    path = '../results/fdemResults/forward_res'
            else:
                path = '../results/fdemResults/forward_res'
                name = 'detection_scenario.png'
            if os.path.exists(path):
                fig.savefig('{}/{}'.format(path, name))
            else:
                os.makedirs(path)
                fig.savefig('{}/{}'.format(path, name))

    def show_mag_map(self, loc_mag, Collection, show=False, save=False, file_name=None, fig=None):
        """

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Empty figure.
        receiver_locations : numpy.ndarray, shape(N*3)
            See fdem_forward_simulation.fdem_forward_simulation receiver_locations.
        mag_data : numpy.ndarray, shape(N*1)
            See fdem_forward_simulation.fdem_forward_simulation mag_data.

        Returns
        -------
        None.
        """
        if fig is not None:
            fig.clf()  # Clear the figure in different detection scene
        else:
            fig = plt.figure()

        receiver_locations = loc_mag[:, 0:3]
        mag_data = loc_mag[:, 3:]
        collection = Collection
        mag_data_total = np.sqrt(np.square(mag_data[:, 0])
                                 + np.square(mag_data[:, 1])
                                 + np.square(mag_data[:, 2]))

        if collection.collection_direction in ["x-axis", "x轴"]:
            mag_data_plotting = mag_data[:, 0]
        elif collection.collection_direction in ["y-axis", "y轴"]:
            mag_data_plotting = mag_data[:, 1]
        elif collection.collection_direction in ["z-axis", "z轴"]:
            mag_data_plotting = mag_data[:, 2]
        elif collection.collection_direction in ["Total", "总场"]:
            mag_data_plotting = mag_data_total

        mag_data_plotting = np.reshape(mag_data_plotting, (1, len(mag_data_plotting)))
        v_max = np.max(mag_data_plotting)
        v_min = np.min(mag_data_plotting)

        ax1 = fig.add_axes([0.13, 0.12, 0.7, 0.8])
        plot2Ddata(
            receiver_locations[:, 0:2],
            mag_data_plotting[0, :],
            ax=ax1,
            ncontour=30,
            clim=(-v_max, v_max),
            contourOpts={"cmap": "bwr"},
        )
        ax1.tick_params(width=0)
        ax1.set_xlabel("x direction [m]")
        ax1.set_ylabel("y direction [m]")
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        ax2 = fig.add_axes([0.85, 0.14, 0.03, 0.76])
        norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
        cbar = mpl.colorbar.ColorbarBase(
            ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr
        )
        cbar.set_label("Secondary field [T]", rotation=270, labelpad=15)
        ax2.tick_params(width=0)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        if show:
            plt.show()
        if save:
            if file_name is not None:
                path = os.path.dirname(file_name)
                name = os.path.basename(file_name)
                if path is '':
                    path = '../results/fdemResults/forward_res'
            else:
                path = '../results/fdemResults/forward_res'
                name = 'mag_map.png'
            if os.path.exists(path):
                fig.savefig('{}/{}'.format(path, name))
            else:
                os.makedirs(path)
                fig.savefig('{}/{}'.format(path, name))

    def show_discretize(self, mesh, mapped_model, Collection, Target, show=False, save=False, file_name=None, fig=None):
        if fig is not None:
            fig.clf()  # Clear the figure in different detection scene
        else:
            fig = plt.figure()
        mesh = mesh
        mapped_model = mapped_model
        collection = Collection
        target_conductivity = Target.conductivity
        ind = int(mesh.hx.size / 2)
        range_x = [collection.x_min, collection.x_max]
        range_y = [-6, 0]

        ax1 = fig.add_axes([0.13, 0.12, 0.7, 0.8])
        mesh.plotSlice(
            mapped_model,
            normal='Y',
            ax=ax1,
            ind=ind,
            grid=True,
            range_x=range_x,
            range_y=range_y
        )
        ax1.tick_params(width=0)
        ax1.set_xlabel("x direction [m]")
        ax1.set_ylabel("z direction [m]")
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        ax2 = fig.add_axes([0.85, 0.12, 0.03, 0.8])
        norm = mpl.colors.Normalize(
            vmin=1e-8, vmax=target_conductivity
        )
        cbar = mpl.colorbar.ColorbarBase(
            ax2, norm=norm, orientation="vertical"
        )
        cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15)
        ax2.tick_params(width=0)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        if show:
            plt.show()
        if save:
            if file_name is not None:
                path = os.path.dirname(file_name)
                name = os.path.basename(file_name)
                if path is '':
                    path = '../results/fdemResults/forward_res'
            else:
                path = '../results/fdemResults/forward_res'
                name = 'discretize.png'
            if os.path.exists(path):
                fig.savefig('{}/{}'.format(path, name))
            else:
                os.makedirs(path)
                fig.savefig('{}/{}'.format(path, name))

    def show_detection_scenario_default(self, fig=None, show=True, save=True):
        """
            in 3D ax show the detection scene,the main part is to show the
            metal cylinder with different posture,the posture of the metal
            cylinder can be showed by the rotation matrix

            Parameters
            ----------
            fig : matplotlib.figure.Figure
                Empty figure.
            target : class Target
                Contains the parameters of the target.
            collection : class Collection
                Contains the parameters of the collection process.

            Returns
            -------
            None.

            """
        if fig is not None:
            fig.clf()  # Clear the figure in different detection scene
        else:
            fig = plt.figure()
        # show the metal cylinder
        target = self.target
        collection = self.collection

        u = np.linspace(0, 2 * np.pi, 50)  # Divide the circle into 20 equal parts
        h = np.linspace(-0.5, 0.5, 2)  # Divide the height(1m)  into 2 equal parts,corresponding to the bottom and top
        x = target.radius * np.sin(u)
        y = target.radius * np.cos(u)

        x = np.outer(x, np.ones(len(h)))  # 20*2
        y = np.outer(y, np.ones(len(h)))  # 20*2
        z = np.outer(np.ones(len(u)), h)  # 20*2
        z = z * target.length

        x_rotation = np.ones(x.shape)  # 旋转后的坐标 20*2
        y_rotation = np.ones(y.shape)
        z_rotation = np.ones(z.shape)

        th1 = target.pitch
        th2 = target.roll
        a = np.array(RotationMatrix(th1, th2, 0))  # 3*3 pitch,roll
        for i in range(2):
            r = np.c_[x[:, i], y[:, i], z[:, i]]  # 20*3
            rT = r @ a  # 20*3
            x_rotation[:, i] = rT[:, 0]
            y_rotation[:, i] = rT[:, 1]
            z_rotation[:, i] = rT[:, 2]
        ax = fig.gca(projection='3d')
        ax.view_init(30, 45)

        ax.plot_surface(x_rotation + target.position[0], y_rotation + target.position[1],
                        z_rotation + target.position[2],
                        color='#E7C261', alpha=1, antialiased=False)
        verts = [list(zip(x_rotation[:, 0] + target.position[0], y_rotation[:, 0] + target.position[1],
                          z_rotation[:, 0] + target.position[2]))]
        ax.add_collection3d(Poly3DCollection(verts, facecolors='#E7C261'))
        verts = [list(zip(x_rotation[:, 1] + target.position[0], y_rotation[:, 1] + target.position[1],
                          z_rotation[:, 1] + target.position[2]))]
        ax.add_collection3d(Poly3DCollection(verts, facecolors='#E7C261'))

        scan_x = np.arange(collection.x_min, collection.x_max + 1e-8, collection.spacing)
        scan_y = np.arange(collection.y_min, collection.y_max + 1e-8, collection.spacing)
        scan_x, scan_y = np.meshgrid(scan_x, scan_y)
        scan_z = np.ones(scan_x.shape) * collection.height

        for i in range(scan_x.shape[1]):
            ax.plot(scan_x[:, i], scan_y[:, i], scan_z[:, i], 'black')
            ax.scatter(scan_x[:, i], scan_y[:, i], scan_z[:, i], marker='o', color='w', edgecolors='blue')
        ax.set_xticks(np.arange(collection.x_min, collection.x_max + 1, 1))
        ax.set_yticks(np.arange(collection.y_min, collection.y_max + 1, 1))
        ax.set_xlabel('X/m')
        ax.set_ylabel('Y/m')
        ax.set_zlabel('Z/m')
        ax.set_xlim(collection.x_min, collection.x_max)
        ax.set_ylim(collection.y_min, collection.y_max)
        ax.set_zlim(target.position[2] - 2, collection.height)

        ax.grid(None)  # delete the background grid
        if save:
            path = './results/fdemResults/{}'.format(self.get_save_fdem_dir())
            if os.path.exists(path):
                path = '{}/{}.png'.format(path, 'detection_scenario')
                fig.savefig(path)
            else:
                os.makedirs(path)
                path = '{}/{}.png'.format(path, 'detection_scenario')
                fig.savefig(path)

        if show:
            plt.show()

    def show_mag_map_default(self, loc_res, fig=None, show=True, save=True):
        """

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Empty figure.
        receiver_locations : numpy.ndarray, shape(N*3)
            See fdem_forward_simulation.fdem_forward_simulation receiver_locations.
        mag_data : numpy.ndarray, shape(N*1)
            See fdem_forward_simulation.fdem_forward_simulation mag_data.

        Returns
        -------
        None.
        """
        if fig is not None:
            fig.clf()
        else:
            fig = plt.figure()
        mag_data = loc_res[:, 3:]
        receiver_locations = loc_res[:, 0:3]
        collection = self.collection
        mag_data_total = np.sqrt(np.square(mag_data[:, 0])
                                 + np.square(mag_data[:, 1])
                                 + np.square(mag_data[:, 2]))

        if collection.collection_direction in ["x-axis", "x轴"]:
            mag_data_plotting = mag_data[:, 0]
        elif collection.collection_direction in ["y-axis", "y轴"]:
            mag_data_plotting = mag_data[:, 1]
        elif collection.collection_direction in ["z-axis", "z轴"]:
            mag_data_plotting = mag_data[:, 2]
        elif collection.collection_direction in ["Total", "总场"]:
            mag_data_plotting = mag_data_total

        mag_data_plotting = np.reshape(mag_data_plotting, (1, len(mag_data_plotting)))
        v_max = np.max(mag_data_plotting)
        v_min = np.min(mag_data_plotting)

        ax1 = fig.add_axes([0.13, 0.12, 0.7, 0.8])
        plot2Ddata(
            receiver_locations[:, 0:2],
            mag_data_plotting[0, :],
            ax=ax1,
            ncontour=30,
            clim=(-v_max, v_max),
            contourOpts={"cmap": "bwr"},
        )
        ax1.tick_params(width=0)
        ax1.set_xlabel("x direction [m]")
        ax1.set_ylabel("y direction [m]")
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        ax2 = fig.add_axes([0.85, 0.14, 0.03, 0.76])
        norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
        cbar = mpl.colorbar.ColorbarBase(
            ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr
        )
        cbar.set_label("Secondary field [T]", rotation=270, labelpad=15)
        ax2.tick_params(width=0)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        if save:
            path = './results/fdemResults/{}'.format(self.get_save_fdem_dir())
            if os.path.exists(path):
                path = '{}/{}.png'.format(path, 'mag_map')
                fig.savefig(path)
            else:
                os.makedirs(path)
                path = '{}/{}.png'.format(path, 'mag_map')
                fig.savefig(path)

        if show:
            plt.show()

    def show_discretize_default(self, mesh, mapped_mode, fig=None, show=True, save=True):
        if fig is not None:
            fig.clf()
        else:
            fig = plt.figure()
        mesh = mesh
        mapped_model = mapped_mode
        collection = self.collection
        target_conductivity = self.target.conductivity
        ind = int(mesh.hx.size / 2)
        range_x = [collection.x_min, collection.x_max]
        range_y = [-6, 0]

        ax1 = fig.add_axes([0.13, 0.12, 0.7, 0.8])
        mesh.plotSlice(
            mapped_model,
            normal='Y',
            ax=ax1,
            ind=ind,
            grid=True,
            range_x=range_x,
            range_y=range_y
        )
        ax1.tick_params(width=0)
        ax1.set_xlabel("x direction [m]")
        ax1.set_ylabel("z direction [m]")
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        ax2 = fig.add_axes([0.85, 0.12, 0.03, 0.8])
        norm = mpl.colors.Normalize(
            vmin=1e-8, vmax=target_conductivity
        )
        cbar = mpl.colorbar.ColorbarBase(
            ax2, norm=norm, orientation="vertical"
        )
        cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15)
        ax2.tick_params(width=0)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        if save:
            path = './results/fdemResults/{}'.format(self.get_save_fdem_dir())
            if os.path.exists(path):
                path = '{}/{}.png'.format(path, 'discretize')
                fig.savefig(path)
            else:
                os.makedirs(path)
                path = '{}/{}.png'.format(path, 'discretize')
                fig.savefig(path)

        if show:
            plt.show()


class TDEMBaseHandler(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    # @abstractmethod
    # def save_forward(self):
    #     pass
    #
    # @abstractmethod
    # def save_cls(self):
    #     pass


class TDEMHandler(TDEMBaseHandler):
    """the data Handler class of TDEM


    Methods
    -------
    - save_response_data:
    - save_sample_data
    - plot_data


    """

    def __init__(self, **kwargs):
        TDEMBaseHandler.__init__(self, **kwargs)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def get_save_tdem_dir(self):
        """
        to get the parameters about this target scene to make file save the data about fdem_forward_simulation

        Returns
        -------
        file_name : str
            the file name to save the data about fdem_forward_simulation and fdem_inversion

        """

        collection = self.collection
        target = self.target
        file_name = "T.material={};T.a=[{:g},{:g}];T.b=[{:g},{:g}];T.a_r_step={:g};T.b_r_step={:g};C.SNR={:g};" \
                    "C.t_split={:g}".format(target.material, target.ta_min, target.ta_max, target.tb_min,
                                            target.tb_max, target.a_r_step, target.b_r_step,
                                            collection.SNR, collection.t_split)
        return file_name

    def save_fwd_data(self, response, file_name=None):
        """save the data of the forward_res
        the function as the general save function of the forward res

        Parameters
        ----------
        response: ndarry
            the dataset to save
        file_name: str
            the name of the file that you want to save, the Default relative
            path is '../results/tdemResults/forward_res',if you just input
            the file name, it will be saved in the path
        """

        response = pd.DataFrame(response)
        path = os.path.dirname(file_name)
        name = os.path.basename(file_name)

        if path is '':
            path = '../results/tdemResults/forward_res'
        if os.path.exists(path):
            response.to_csv('{}/{}'.format(path, name))
        else:
            os.makedirs(path)
            response.to_csv('{}/{}'.format(path, name))

    def save_sample_data(self, sample, file_name, show=False):
        """save the sample data of the forward_res

        Parameters
        ----------
        sample: ndarry
            the dataset to save
        file_name: str
            the name of the file that you want to save, the Default relative
            path is '../results/tdemResults/forward_res',if you just input
            the file name, it will be saved in the path
        show: bool
            show the picture of the sample data
        """
        sample_data = sample['data']
        path = os.path.dirname(file_name)
        name = os.path.basename(file_name)
        pic_name = name.replace('.csv', '.png')

        if path is '' or None:
            path = '../results/tdemResults/forward_res'
        if os.path.exists(path):
            sample_data.to_csv('{}/{}'.format(path, name))
            self.plot_data(sample['M1'], sample['M2'], sample['M1_without_noise'],
                           sample['M2_without_noise'], sample['t'], sample['SNR'], sample['material'],
                           sample['ta'], sample['tb'], '{}/{}'.format(path, pic_name), show)
        else:
            os.makedirs(path)
            sample_data.to_csv('{}/{}'.format(path, name))
            self.plot_data(sample['M1'], sample['M2'], sample['M1_without_noise'],
                           sample['M2_without_noise'], sample['t'], sample['SNR'], sample['material'],
                           sample['ta'], sample['tb'], '{}/{}'.format(path, pic_name), show)

    def save_fwd_data_default(self, response):
        """save the data of the forward_res
        the function as the general save function of the forward res

        Parameters
        ----------
        response: ndarry
            the dataset to save
        file_name: str
            the name of the file that you want to save, the Default relative
            path is '../results/tdemResults/forward_res',if you just input
            the file name, it will be saved in the path
        """

        response = pd.DataFrame(response)
        file_name = self.get_save_tdem_dir()
        path = './results/tdemResults/{}/originData'.format(file_name)

        if os.path.exists(path):
            response.to_csv('{}/response.csv'.format(path))
        else:
            os.makedirs(path)
            response.to_csv('{}/response.csv'.format(path))

    def save_sample_data_default(self, sample, fig=None, show=True, save=True):
        """save the sample data of the forward_res

        Parameters
        ----------
        sample: ndarry
            the dataset to save
        file_name: str
            the name of the file that you want to save, the Default relative
            path is '../results/tdemResults/forward_res',if you just input
            the file name, it will be saved in the path
        show: bool
            show the picture of the sample data
        """
        if fig is not None:
            fig.clf()
        else:
            fig = plt.figure()
        sample_data = sample['data']
        file_name = self.get_save_tdem_dir()
        snr = sample['SNR']
        if save:
            path = './results/tdemResults/{}/originData/sample selected'.format(file_name)
            if os.path.exists(path):
                sample_data.to_csv('{}/sample_{}dB.csv'.format(path, snr))
                self.plot_data(sample['M1'], sample['M2'], sample['M1_without_noise'],
                               sample['M2_without_noise'], sample['t'], sample['SNR'], sample['material'],
                               sample['ta'], sample['tb'], '{}/sample_{}dB.png'.format(path, snr), show, fig)
            else:
                os.makedirs(path)
                sample_data.to_csv('{}/sample_{}dB.csv'.format(path, snr))
                self.plot_data(sample['M1'], sample['M2'], sample['M1_without_noise'],
                               sample['M2_without_noise'], sample['t'], sample['SNR'], sample['material'],
                               sample['ta'], sample['tb'], '{}/sample_{}dB.png'.format(path, snr), show, fig)
        else:
            self.plot_data(sample['M1'], sample['M2'], sample['M1_without_noise'],
                           sample['M2_without_noise'], sample['t'], sample['SNR'], sample['material'],
                           sample['ta'], sample['tb'], None, show, fig)

    def plot_data(self, M1, M2, M1_without_noise, M2_without_noise, t, SNR, material, ta, tb, file_name=None,
                  show=False, fig=None):
        """show the sample data

        Parameters
        ----------
        M1: ndarry
            the M1 response of the target
        M2: ndarry
            the M2 response of the target
        M1_without_noise: ndarry
            the M1 response of the target without the noise
        M2_without_noise: ndarry
            the M2 response of the target without the noise
        t: int
            the times collected per second
        SNR: int
            SNR(Signal to Noise Ratio)
        material: str
            the material of the target of the sample
        ta: int
            the radial radius of the target of the sample
        tb: int
            the axial radius of the target of the sample
        file_name: str
            the name of the file that you want to save, the Default relative
            path is '../results/tdemResults/forward_res',if you just input
            the file name, it will be saved in the path
        show: bool
            whether to show the picture
        """
        # fig, ax = plt.subplots()
        if fig is not None:
            # ax = fig.add_subplot()
            ax = fig.add_axes([0.23, 0.12, 0.7, 0.8])
        else:
            fig, ax = plt.subplots()

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-8, 1e1)
        ax.plot(t, np.array(M1_without_noise).flatten(), '--', color="r", label="M1_noiseless")
        ax.plot(t, np.array(M2_without_noise).flatten(), '--', color="b", label="M2_noiseless")
        ax.plot(t, M1, 'o', color="y", label="M1")
        ax.plot(t, M2, 'o', color="g", label="M2")
        ax.set_xlabel("t /s")
        ax.set_ylabel("M")
        ax.set_title(str(material) + " ta=" + "%.2f" % ta + " tb=" + "%.2f" % tb + " SNR=" + str(SNR) + "dB")
        if file_name is not None:
            fig.savefig(file_name, dpi=1000, bbox_inches='tight')
        if show:
            fig.show()

    def save_preparation_default(self, train_set, test_set, task):
        if task == 'material':
            train_set_material = pd.DataFrame(train_set)
            test_set_material = pd.DataFrame(test_set)
            path_material = './results/tdemResults/{}/prepareData_material'.format(self.get_save_tdem_dir())
            if os.path.exists(path_material):
                train_set_material.to_csv('{}/train_set.csv'.format(path_material))
                test_set_material.to_csv('{}/test_set.csv'.format(path_material))
            else:
                os.makedirs(path_material)
                train_set_material.to_csv('{}/train_set.csv'.format(path_material))
                test_set_material.to_csv('{}/test_set.csv'.format(path_material))
        if task == 'shape':
            train_set_shape = pd.DataFrame(train_set)
            test_set_shape = pd.DataFrame(test_set)
            path_shape = './results/tdemResults/{}/prepareData_shape'.format(self.get_save_tdem_dir())
            if os.path.exists(path_shape):
                train_set_shape.to_csv('{}/train_set.csv'.format(path_shape))
                test_set_shape.to_csv('{}/test_set.csv'.format(path_shape))
            else:
                os.makedirs(path_shape)
                train_set_shape.to_csv('{}/train_set.csv'.format(path_shape))
                test_set_shape.to_csv('{}/test_set.csv'.format(path_shape))

    def save_dim_reduction_default(self, train_set, test_set, task, dim_method):
        train_set = train_set
        test_set = test_set
        task = task
        dim_red_method = dim_method

        if task == 'material':
            train_set_material = pd.DataFrame(train_set)
            test_set_material = pd.DataFrame(test_set)
            path_material = './results/tdemResults/{}/dimReductionData_material/{}'.format(self.get_save_tdem_dir(),
                                                                                           dim_red_method)
            if os.path.exists(path_material):
                train_set_material.to_csv('{}/train_set.csv'.format(path_material))
                test_set_material.to_csv('{}/test_set.csv'.format(path_material))
            else:
                os.makedirs(path_material)
                train_set_material.to_csv('{}/train_set.csv'.format(path_material))
                test_set_material.to_csv('{}/test_set.csv'.format(path_material))
        if task == 'shape':
            train_set_material = pd.DataFrame(train_set)
            test_set_material = pd.DataFrame(test_set)
            path_shape = './results/tdemResults/{}/dimReductionData_shape/{}'.format(self.get_save_tdem_dir(),
                                                                                     dim_red_method)
            if os.path.exists(path_shape):
                train_set_material.to_csv('{}/train_set.csv'.format(path_shape))
                test_set_material.to_csv('{}/test_set.csv'.format(path_shape))
            else:
                os.makedirs(path_shape)
                train_set_material.to_csv('{}/train_set.csv'.format(path_shape))
                test_set_material.to_csv('{}/test_set.csv'.format(path_shape))

    def plot_confusion_matrix(self, cls_res, type, show=False, save=False, file_name=None):
        """This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Parameters:
        -----------
        type: list
            the list of classification types
        show: bool
            whether to show
        save: bool
            whether to save
        file_name: str
            the name of the file that you want to save, the Default relative
            path is '../results/tdemResults/classify_res',if you just input
            the file name, it will be saved in the path, if you don't input
            the file name, it will be saved by the name 'cls_res.pdf'

        """
        cmap = plt.cm.YlGnBu
        plt.figure()
        cm = confusion_matrix(y_true=cls_res['y_true'], y_pred=cls_res['y_pred'])
        classes = type
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.ylabel('True label', fontsize=17)
        plt.xlabel('Predicted label', fontsize=17)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Confusion matrix', fontsize=17)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0, fontsize=15)
        plt.yticks(tick_marks - 0.25, classes, rotation=90, fontsize=15)
        plt.tick_params(bottom=False, top=False, left=False, right=False)  # 移除全部刻度线
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=26)

        plt.colorbar(shrink=1)
        if show:
            plt.show()
        if save:
            if file_name is not None:
                path = os.path.dirname(file_name)
                name = os.path.basename(file_name)
                if path is '':
                    path = '../results/tdemResults/classify_res'
            else:
                path = '../results/tdemResults/classify_res'
                name = 'cls_res.pdf'
            if os.path.exists(path):
                plt.savefig('{}/{}'.format(path, name), format='pdf', bbox_inches='tight', dpi=600)
            else:
                os.makedirs(path)
                plt.savefig('{}/{}'.format(path, name), format='pdf', bbox_inches='tight', dpi=600)

    # cm：混淆矩阵；classes：类别名称
    def plot_confusion_matrix_default(self, cls_result, task, fig=None, show=False, save=True):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if fig is not None:
            # ax = fig.add_subplot()
            fig.clf()
            ax = fig.add_axes([0.23, 0.12, 0.7, 0.8])
        else:
            fig = plt.figure()
            ax = fig.add_axes([0.23, 0.12, 0.7, 0.8])
        cmap = plt.cm.YlGnBu
        cm = confusion_matrix(y_true=cls_result['y_true'], y_pred=cls_result['y_pred'])
        if task == 'material':
            classes = self.target.material
        else:
            classes = self.target.shape
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        ax.set_ylabel('True label', fontsize=12)
        ax.set_xlabel('Predicted label', fontsize=12)
        ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title('Confusion matrix', fontsize=12)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=0, fontsize=8)
        ax.set_yticklabels(classes, rotation=360, fontsize=8)
        # ax.set_xticks(tick_marks, classes)
        # plt.xticks(tick_marks, classes, rotation=0, fontsize=15)
        # plt.yticks(tick_marks - 0.25, classes, rotation=90, fontsize=15)
        # ax.set_yticks(3 - 0.25, classes)
        # plt.tick_params(bottom=False, top=False, left=False, right=False)  # 移除全部刻度线
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=26)

        # fig.colorbar(shrink=1)
        # fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax)

        if save:
            path = './results/tdemResults/{}'.format(self.get_save_tdem_dir())
            if os.path.exists(path):
                path = '{}/{}.pdf'.format(path, 'cls_res')
                fig.savefig(path, format='pdf', bbox_inches='tight', dpi=600)
            else:
                os.makedirs(path)
                path = '{}/{}.pdf'.format(path, 'cls_res')
                fig.savefig(path, format='pdf', bbox_inches='tight', dpi=600)
        if show:
            plt.show()

    def save_cls_res(self, cls_res, file_name):
        """save the classification result

        Parameters
        ----------
        cls_res: dict
            the res of the classification, conclude the accuracy, the true value and
            the pred value
        file_name: str
            the name of the file that you want to save, the Default relative
            path is '../results/tdemResults/classify_res',if you just input
            the file name, it will be saved in the path

        """

        y_true = np.array(cls_res['y_true'], dtype=np.int)
        y_pred = np.array(cls_res['y_pred'], dtype=np.int)
        y = np.c_[y_true, y_pred]
        accuracy = np.zeros(shape=(y_true.shape[0], 1), dtype=np.str)

        res = np.c_[y, accuracy]
        res = pd.DataFrame(res, columns=['y_true', 'y_pred', 'accuracy'])
        res.iloc[0, 2] = cls_res['accuracy']
        if file_name is not None:
            path = os.path.dirname(file_name)
            name = os.path.basename(file_name)
            if path is '':
                path = '../results/tdemResults/classify_res'
        else:
            path = '../results/tdemResults/classify_res'
            name = 'cls_res.csv'

        if os.path.exists(path):
            res.to_csv('{}/{}'.format(path, name))
        else:
            os.makedirs(path)
            res.to_csv('{}/{}'.format(path, name))

    def save_cls_res_default(self, cls_res):
        """save the classification result defaultly

        Parameters
        ----------
        cls_res: dict
            the res of the classification, conclude the accuracy, the true value and
            the pred value
        file_name: str
            the name of the file that you want to save, the Default relative
            path is '../results/tdemResults/classify_res',if you just input
            the file name, it will be saved in the path

        """

        y_true = np.array(cls_res['y_true'], dtype=np.int)
        y_pred = np.array(cls_res['y_pred'], dtype=np.int)
        y = np.c_[y_true, y_pred]
        accuracy = np.zeros(shape=(y_true.shape[0], 1), dtype=np.str)

        res = np.c_[y, accuracy]
        res = pd.DataFrame(res, columns=['y_true', 'y_pred', 'accuracy'])
        res.iloc[0, 2] = cls_res['accuracy']
        path = './results/tdemResults/{}'.format(self.get_save_tdem_dir())
        if os.path.exists(path):
            res.to_csv('{}/{}'.format(path, 'cls_res.csv'))
        else:
            os.makedirs(path)
            res.to_csv('{}/{}'.format(path, 'cls_res.csv'))
