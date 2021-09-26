# -*- coding: utf-8 -*-
"""
The target class, represent the target in underground detection

Class:
- Target: the target class in TDEM
"""
__all__ = ['Target']


class Target(object):
    """Refer in particular to spheroid targets.

    Attributes
    ----------
    material: list
        the list of material of the target
    shape: list
        the list of shape of the target
    attribute: 2D-ndarray
        the relative permeability, permeability of vacuum and conductivity of the target
    ta_min : float
        represent the min radial radius of the spheroid targets
    ta_max : float
        represent the max radial radius of the spheroid targets
    tb_min : float
        represent the min axial radius of the spheroid targets
    tb_max : float
        represent the max axial radius of the spheroid targets
    a_r_step: float
        represent the changed step of radial radius
    b_r_step: float
        represent the changed step of axial radius
    kwgs: dict
        the extensible attribute

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
        for key, val in kwargs.items():
            setattr(self, key, val)
