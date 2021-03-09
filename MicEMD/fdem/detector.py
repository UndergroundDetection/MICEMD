"""

"""
__all__ = ['Detector']

import numpy as np


class Detector(object):
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
