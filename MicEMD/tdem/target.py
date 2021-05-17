# -*- coding: utf-8 -*-
__all__ = ['Target']

import numpy as np
from scipy.constants import mu_0

steel_attribute = np.array([[696.3028547, 875 * 1e-6, 50000000]])
ni_attribute = np.array([[99.47183638, 125 * 1e-6, 14619883.04]])
al_attribute = np.array([[1.000022202, 1.256665 * 1e-6, 37667620.91]])
material_dict = {'Steel': steel_attribute, 'Ni': ni_attribute, 'Al': al_attribute}
material_list = ['Steel', 'Ni', 'Al']

class Target(object):
    """
    Refer in particular to spheroid targets.

    Attributes
    ----------
    attribute: 2D ndarray
        conclude the relative permeability, permeability of vacuum and conductivity of the target
    ta : float
        represent the radial radius of the spheroid targets
    tb : float
        represent the axial radius of the spheroid targets
    r_step: float
        represent the changed step of radial adn axial radius

    """

    def __init__(self, material, shape, attribute, ta_min, ta_max, tb_min, tb_max, a_r_step, b_r_step, **kwargs):
        self.material = material
        self.shape = shape
        self.attribute = attribute
        self.ta_min = ta_min
        self.ta_max = ta_max
        self.tb_min = tb_min
        self.tb_max = tb_max
        self.a_r_step = a_r_step
        self.b_r_step = b_r_step

