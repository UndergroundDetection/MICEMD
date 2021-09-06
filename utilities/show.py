# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:17:51 2020

"""
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from MicEMD.utils import RotationMatrix


def show_fdem_detection_scenario(fig, target, collection):
    """In 3D ax show the detection scene,the main part is to show the
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

    ax.plot_surface(x_rotation + target.position[0], y_rotation + target.position[1], z_rotation + target.position[2],
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


# import matplotlib.pyplot as plt
# import MicEMD.fdem as f
# fig1 = plt.figure()
# fig2 = plt.figure()
# target = f.Target(5.71e7, 1.26e-6, 0.2, 0, 0, 1, 0, 0, -5)
# detector = f.Detector(0.4, 20, 1000, 0, 0)
# collection = f.Collection(0.4, 0.1, 30, -2, 2, -2, 2, 'z-axis')
# show_fdem_detection_scenario(fig1, target, collection)
# forward_result = f.simulate(target, detector, collection, 'simpeg', True)
# mag_data = forward_result.mag_data
# mag_data_total = np.sqrt(np.square(mag_data[:, 0])
#                                  + np.square(mag_data[:, 1])
#                                  + np.square(mag_data[:, 2]))
#
# if collection.collection_direction in ["x-axis", "x轴"]:
#     mag_data_plotting = mag_data[:, 0]
# elif collection.collection_direction in ["y-axis", "y轴"]:
#     mag_data_plotting = mag_data[:, 1]
# elif collection.collection_direction in ["z-axis", "z轴"]:
#     mag_data_plotting = mag_data[:, 2]
# elif collection.collection_direction in ["Total", "总场"]:
#     mag_data_plotting = mag_data_total
#
# show_fdem_mag_map(fig2, forward_result.receiver_locations, mag_data_plotting)
#
# plt.show()
