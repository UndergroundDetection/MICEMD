# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:17:51 2020

@author: Liu Wen
"""

import numpy as np

from SimPEG.utils import plot2Ddata

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


def show_fdem_detection_scenario(fig, target, collection):
    """

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

    fig.clf()
    detection_deep = 5


def show_fdem_mag_map(fig, receiver_locations, mag_data):
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

    mag_data_plotting = np.reshape(mag_data, (1, len(mag_data)))
    v_max = np.max(mag_data_plotting)
    v_min = np.min(mag_data_plotting)

    ax1 = fig.add_axes([0.11, 0.12, 0.75, 0.8])
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

    ax2 = fig.add_axes([0.87, 0.12, 0.03, 0.8])
    norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr
    )
    cbar.set_label("Secondary field [T]", rotation=270, labelpad=15)
    ax2.tick_params(width=0)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))


def show_discretize(fig, mesh, mapped_model, normal, ind, range_x, range_y,
                    target_conductivity):

    fig.clf()

    ax1 = fig.add_axes([0.13, 0.12, 0.7, 0.8])
    mesh.plotSlice(
        mapped_model,
        normal=normal,
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
