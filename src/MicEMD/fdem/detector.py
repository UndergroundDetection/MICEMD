# -*- coding: utf-8 -*-
"""
The detector class, represent the detector in FDEM

Class:
- Detector: the detector class in FDEM
"""
__all__ = ['Detector']

import numpy as np


class Detector(object):
    """the detector of the FDEM

    Attributes
    ----------
    radius: float
        the radius of the detector
    current: float
        the current of the detector
    frequency: float
        the frequency of the detector
    pitch: float
        the pitch angle of the detector
    roll: float
        the roll angle of the detector

    Methods:
    mag_moment:
        return the magnetic moment value of transmitter coil

    """

    def __init__(self, radius, current, frequency, pitch, roll):
        self.radius = radius
        self.current = current
        self.frequency = frequency
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
