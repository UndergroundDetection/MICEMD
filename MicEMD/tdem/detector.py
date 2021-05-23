# -*- coding: utf-8 -*-
"""
The detector class, represent the detector in TDEM

Class:
- Detector: the detector class in TDEM
"""
__all__ = ['Detector']

import numpy as np


class Detector(object):
    """the detector in TDEM

    Attributes
    ----------
    radius: float
        the radius of the detector
    current: float
        the current of the detector
    pitch: float
        the pitch angle of the target
    roll: float
        the roll angle of the target

    Methods:
    ----------
    mag_moment
        Returns the magnetic moment value of transmitter coil
    """

    def __init__(self, radius, current, pitch, roll):
        self.radius = radius
        self.current = current
        self.pitch = pitch
        self.roll = roll

    @property
    def mag_moment(self):
        """
        Calculate magnetic moment value of transmitter coil according to
        detector parameters.

        Returns
        -------
        TYPE : float
            The value of magnetic moment of z axis.
        """

        return self.current * np.pi * self.radius ** 2
