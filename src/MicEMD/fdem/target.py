# -*- coding: utf-8 -*-
"""
The target class, represent the target in underground detection

Class:
- Target: the target class in FDEM
"""
__all__ = ['Target']

import numpy as np
from scipy.constants import mu_0


class Target(object):
    """Refer in particular to cylinder targets.

    Attributes
    ----------
    conductivity: float
        the conductivity of the target
    permeability: float
        the permeability of the target
    radius: float
        The radius of the base of a cylinder
    pitch: float
        the pitch angle of the cylinder
    roll: float
        the roll angle of the cylinder
    length: float
        the length of the target
    position: list
        the position of the target, [x, y, z]
    kwgs: dict
        the extensible attribute

    Methods
    -------
    get_principal_axis_polarizability
        return the principal axis polarizabilities of the target

    """

    def __init__(self, conductivity, permeability, radius, pitch, roll,
                 length, position_x, position_y, position_z, **kwargs):
        self.conductivity = conductivity
        self.permeability = permeability
        self.radius = radius
        self.pitch = pitch
        self.roll = roll
        self.length = length
        self.position = [position_x, position_y, position_z]
        for key, val in kwargs.items():
            setattr(self, key, val)

    def get_principal_axis_polarizability(self, frequency):
        """
        Calculate the principal axis polarizabilities of the target according
        to the target's parameters and the frequency of primary field.

        Parameters
        ----------
        frequency : float
            Frequency of the primary field.

        Returns
        -------
        polarizability : numpy.array, size=3

        References
        ----------
        Wan Y, Wang Z, Wang P, et al. A Comparative Study of Inversion
        Optimization Algorithms for Underground Metal Target Detection[J].
        IEEE Access, 2020, 8: 126401-126413. equation (9)(10).
        """

        volume = np.pi * self.radius ** 2 * self.length
        omega = 2 * np.pi * frequency
        relative_permeability = self.permeability / mu_0
        tau = (self.radius ** 2 * self.conductivity * mu_0
               / relative_permeability ** 2)
        beta_x = beta_y = 0.5 * volume * (0.35
                                          + (np.sqrt(1j * omega * tau) - 2)
                                          / (np.sqrt(1j * omega * tau) + 1))

        beta_z = (self.length * volume / (4 * self.radius)
                  * (- 0.7
                     + (np.sqrt(1j * omega * tau * 31) - 2)
                     / (np.sqrt(1j * omega * tau * 31) + 1)))

        return np.array([abs(beta_x), abs(beta_y), abs(beta_z)])
    def get_principal_axis_polarizability_complex(self, frequency):
        """
        Calculate the principal axis polarizabilities of the target according
        to the target's parameters and the frequency of primary field.

        Parameters
        ----------
        frequency : float
            Frequency of the primary field.

        Returns
        -------
        polarizability : numpy.array, size=3

        References
        ----------
        Wan Y, Wang Z, Wang P, et al. A Comparative Study of Inversion
        Optimization Algorithms for Underground Metal Target Detection[J].
        IEEE Access, 2020, 8: 126401-126413. equation (9)(10).
        """

        volume = np.pi * self.radius ** 2 * self.length
        omega = 2 * np.pi * frequency
        relative_permeability = self.permeability / mu_0
        # relative_permeability = 1
        tau = self.radius ** 2 * self.conductivity * mu_0 / relative_permeability ** 2
        z1 = np.sqrt(1j * omega * tau)-2
        z2 = np.sqrt(1j * omega * tau)+1

        z0 = z1/z2
        z01 = np.sqrt(1j * omega * tau * 31) - 2
        z02 = np.sqrt(1j * omega * tau * 31) + 1
        z00 = z01 / z02
        beta_x = beta_y = 0.5 * volume * (0.05 + z0)

        beta_z = self.length * volume / (4 * self.radius) * (- 0.1 + z00)

        return np.array([beta_x, beta_y, beta_z])
