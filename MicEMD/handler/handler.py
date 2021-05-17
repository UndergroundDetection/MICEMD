__all__ = ['FDEMHandler', 'TDEMHandler']

from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import pandas as pd
import os
import itertools

from matplotlib import gridspec
from sklearn.metrics import confusion_matrix

from ..preprocessor import data_prepare
from ..utils import RotationMatrix
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from SimPEG.utils import plot2Ddata
import matplotlib as mpl
import matplotlib.pyplot as plt


class FDEMBaseHandler(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, ForwardResult, InvResult):
        self.forward_result = ForwardResult
        self.inv_result = InvResult

    @abstractmethod
    def save_forward(self):
        pass

    @abstractmethod
    def save_inv(self):
        pass


class FDEMHandler(FDEMBaseHandler):
    def __init__(self, ForwardResult, InvResult):
        FDEMBaseHandler.__init__(self, ForwardResult, InvResult)

    def get_save_fdem_dir(self):
        """
        to get the parameters about this target scene to make file save the data about fdem_forward_simulation

        Returns
        -------
        file_name : str
            the file name to save the data about fdem_forward_simulation and fdem_inversion

        """
        simulation = self.forward_result.simulation
        collection = simulation.model.survey.source.collection
        target = simulation.model.survey.source.target
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

    def save_mag_data(self, file_name):
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

        fdem_mag_data = self.forward_result.mag_data
        fdem_receiver_locs = self.forward_result.receiver_locations

        mag_data_index = [0] * (fdem_mag_data.shape[0])
        for i in range(fdem_mag_data.shape[0]):
            mag_data_index[i] = 'the ' + str(i + 1) + ' observation point'
        data = np.c_[fdem_receiver_locs, fdem_mag_data]
        mag_data = pd.DataFrame(data, columns=['x', 'y', 'z', 'hx', 'hy', 'hz'], index=mag_data_index)

        path = './results/fdemResults/{}'.format(file_name)

        if os.path.exists(path):
            mag_data.to_csv('{}/mag_data.csv'.format(path))
        else:
            os.makedirs(path)
            mag_data.to_csv('{}/mag_data.csv'.format(path))

    def save_result(self, file_name):
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
        fdem_true_properties = self.inv_result.true_parameters
        fdem_estimate_properties = self.inv_result.estimate_parameters
        fdem_estimate_error = self.inv_result.error
        path = './results/fdemResults/{}'.format(file_name)
        invResult_index = ['True_value', 'Estimate_value', 'Error']
        property = np.vstack(([fdem_true_properties, fdem_estimate_properties, fdem_estimate_error]))

        inv_result = pd.DataFrame(property,
                                  columns=['x', 'y', 'z', 'polarizability_1', 'polarizability_2', 'polarizability_3',
                                           'pitch', 'roll'],
                                  index=invResult_index)
        inv_filename = self.inv_result.condition['method'] + '_invResult'

        if os.path.exists(path):
            inv_result.to_csv('{}/{}.csv'.format(path, inv_filename))
        else:
            os.makedirs(path)
            inv_result.to_csv('{}/{}.csv'.format(path, inv_filename))

    def save_forward(self, forward_flag):
        if forward_flag:
            self.save_mag_data(self.get_save_fdem_dir())

    def save_inv(self, inv_flag):
        if inv_flag:
            self.save_result(self.get_save_fdem_dir())

    def show_fdem_detection_scenario(self, fig):
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
        fig.clf()  # Clear the figure in different detection scene
        # show the metal cylinder
        target = self.forward_result.simulation.model.survey.source.target
        collection = self.forward_result.simulation.model.survey.source.collection

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
        path = './results/fdemResults/{}'.format(self.get_save_fdem_dir())
        path = '{}/{}.png'.format(path, 'detection_scenario')
        fig.savefig(path)

    def show_fdem_mag_map(self, fig):
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

        fig.clf()
        mag_data = self.forward_result.mag_data
        receiver_locations = self.forward_result.receiver_locations
        collection = self.forward_result.simulation.model.survey.source.collection
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

        path = './results/fdemResults/{}'.format(self.get_save_fdem_dir())
        path = '{}/{}.png'.format(path, 'mag_map')
        fig.savefig(path)

    def show_discretize(self, fig):
        fig.clf()
        mesh = self.forward_result.mesh
        mapped_model = self.forward_result.mapped_model
        collection = self.forward_result.simulation.model.survey.source.collection
        target_conductivity = self.forward_result.simulation.model.survey.source.target.conductivity
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
        path = './results/fdemResults/{}'.format(self.get_save_fdem_dir())
        path = '{}/{}.png'.format(path, 'discretize')
        fig.savefig(path)


class TDEMBaseHandler(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, ForwardResult, PreResult, ClsResult):
        self.forward_result = ForwardResult
        self.pre_result = PreResult
        self.cls_result = ClsResult

    @abstractmethod
    def save_forward(self):
        pass

    @abstractmethod
    def save_cls(self):
        pass


class TDEMHandler(TDEMBaseHandler):
    def __init__(self, ForwardResult, PreResult=None, ClsResult=None):
        TDEMBaseHandler.__init__(self, ForwardResult, PreResult, ClsResult)
        self.forward_result = ForwardResult
        self.pre_result = PreResult
        self.cls_result = ClsResult

    def save_cls(self, inv_flag):
        if inv_flag:
            pass

    def get_save_fdem_dir(self):
        """
        to get the parameters about this target scene to make file save the data about fdem_forward_simulation

        Returns
        -------
        file_name : str
            the file name to save the data about fdem_forward_simulation and fdem_inversion

        """
        simulation = self.forward_result.simulation
        collection = simulation.model.survey.source.collection
        target = simulation.model.survey.source.target
        file_name = "T.material={};T.a=[{:g},{:g}];T.b=[{:g},{:g}];T.a_r_step={:g};T.b_r_step={:g};C.SNR={:g};" \
                    "C.t_split={:g}".format(target.material, target.ta_min, target.ta_max, target.tb_min,
                                            target.tb_max, target.a_r_step, target.b_r_step,
                                            collection.SNR, collection.t_split)
        return file_name

    def save_response_data(self, file_name):
        response = self.forward_result.response
        response = pd.DataFrame(response)
        path = './results/tdemResults/{}/originData'.format(file_name)

        if os.path.exists(path):
            response.to_csv('{}/response.csv'.format(path))
        else:
            os.makedirs(path)
            response.to_csv('{}/response.csv'.format(path))

    def save_sample_data(self, file_name, show):
        sample = self.forward_result.sample

        sample_data = sample['data']
        snr = sample['SNR']
        path = './results/tdemResults/{}/originData/sample selected'.format(file_name)
        if os.path.exists(path):
            sample_data.to_csv('{}/sample_{}dB.csv'.format(path, snr))
            self.plot_data(sample['M1'], sample['M2'], sample['M1_without_noise'],
                           sample['M2_without_noise'], sample['t'], sample['SNR'], sample['material'],
                           sample['ta'], sample['tb'], '{}/sample_{}dB.png'.format(path, snr), show)
        else:
            os.makedirs(path)
            sample_data.to_csv('{}/sample_{}dB.csv'.format(path, snr))
            self.plot_data(sample['M1'], sample['M2'], sample['M1_without_noise'],
                           sample['M2_without_noise'], sample['t'], sample['SNR'], sample['material'],
                           sample['ta'], sample['tb'], '{}/sample_{}dB.png'.format(path, snr), show)

    def save_forward(self, save, show):
        if save:
            self.save_response_data(self.get_save_fdem_dir())
            self.save_sample_data(self.get_save_fdem_dir(), show)

    def plot_data(self, M1, M2, M1_without_noise, M2_without_noise, t,
                  SNR, material, ta, tb, file_name, show):
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-8, 1e1)
        plt.plot(t, np.array(M1_without_noise).flatten(), '--', color="r", label="M1_noiseless")
        plt.plot(t, np.array(M2_without_noise).flatten(), '--', color="b", label="M2_noiseless")
        plt.plot(t, M1, 'o', color="y", label="M1")
        plt.plot(t, M2, 'o', color="g", label="M2")
        plt.xlabel("t /s")
        plt.ylabel("M")
        plt.title(str(material) + " ta=" + "%.2f" % ta + " tb=" + "%.2f" % tb + " SNR=" + str(SNR) + "dB")
        plt.savefig(file_name, dpi=1000, bbox_inches='tight')
        if show:
            plt.show()
    def save_preparation(self, data_set, task, save_flag):
        if save_flag and task == 'material':
            train_set_material = pd.DataFrame(data_set[0])
            test_set_material = pd.DataFrame(data_set[1])
            path_material = './results/tdemResults/{}/prepareData_material'.format(self.get_save_fdem_dir())
            if os.path.exists(path_material):
                train_set_material.to_csv('{}/train_set.csv'.format(path_material))
                test_set_material.to_csv('{}/test_set.csv'.format(path_material))
            else:
                os.makedirs(path_material)
                train_set_material.to_csv('{}/train_set.csv'.format(path_material))
                test_set_material.to_csv('{}/test_set.csv'.format(path_material))
        if save_flag and task == 'shape':
            train_set_shape = pd.DataFrame(data_set[0])
            test_set_shape = pd.DataFrame(data_set[1])
            path_shape = './results/tdemResults/{}/prepareData_shape'.format(self.get_save_fdem_dir())
            if os.path.exists(path_shape):
                train_set_shape.to_csv('{}/train_set.csv'.format(path_shape))
                test_set_shape.to_csv('{}/test_set.csv'.format(path_shape))
            else:
                os.makedirs(path_shape)
                train_set_shape.to_csv('{}/train_set.csv'.format(path_shape))
                test_set_shape.to_csv('{}/test_set.csv'.format(path_shape))

    def save_dim_reduction(self, saveFlag):
        train_set = self.pre_result.train_set
        test_set = self.pre_result.test_set
        task = self.pre_result.task
        dim_red_method = self.pre_result.dim_red_method

        if saveFlag and task == 'material':
            train_set_material = pd.DataFrame(train_set)
            test_set_material = pd.DataFrame(test_set)
            path_material = './results/tdemResults/{}/dimReductionData_material/{}'.format(self.get_save_fdem_dir(),
                                                                                           dim_red_method)
            if os.path.exists(path_material):
                train_set_material.to_csv('{}/train_set.csv'.format(path_material))
                test_set_material.to_csv('{}/test_set.csv'.format(path_material))
            else:
                os.makedirs(path_material)
                train_set_material.to_csv('{}/train_set.csv'.format(path_material))
                test_set_material.to_csv('{}/test_set.csv'.format(path_material))
        if saveFlag and task == 'shape':
            train_set_material = pd.DataFrame(train_set)
            test_set_material = pd.DataFrame(test_set)
            path_shape = './results/tdemResults/{}/dimReductionData_shape/{}'.format(self.get_save_fdem_dir(),
                                                                                     dim_red_method)
            if os.path.exists(path_shape):
                train_set_material.to_csv('{}/train_set.csv'.format(path_shape))
                test_set_material.to_csv('{}/test_set.csv'.format(path_shape))
            else:
                os.makedirs(path_shape)
                train_set_material.to_csv('{}/train_set.csv'.format(path_shape))
                test_set_material.to_csv('{}/test_set.csv'.format(path_shape))

    # cm：混淆矩阵；classes：类别名称
    def plot_confusion_matrix(self, show):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        cmap = plt.cm.YlGnBu
        plt.figure()
        # gs = gridspec.GridSpec(1, 2, width_ratios=[4.45, 5.55])
        # plt.subplot(gs[0])
        cm = confusion_matrix(y_true=self.cls_result.cls_result[1], y_pred=self.cls_result.cls_result[2])
        if self.pre_result.task == 'material':
            classes = self.forward_result.simulation.model.survey.source.target.material
        else:
            classes = self.forward_result.simulation.model.model.survey.source.target.shape
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
