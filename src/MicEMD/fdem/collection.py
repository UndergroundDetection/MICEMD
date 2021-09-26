# -*- coding: utf-8 -*-
"""
The Collection class, conclude the parameters about collecting in underground detection system

Class:
- Collection: the class conclude the parameters about collecting in FDEM
"""
__all__ = ['Collection']


import numpy as np
from ..utils import mkvc


class Collection(object):
    """

    Attributes
    ----------
    spacing: float
        the spacing of collection in x and y axis
    height: float
        The height of the detector above the ground
    SNR: float
        the SNR of the environment
    x_min: float
        the min value of the x axis collection
    x_max: float
        the max value of the x axis collection
    y_min: float
        the min value of the y axis collection
    y_max: float
        the max value of the y axis collection
    collection_direction: str
        the direction of the collection magnetic field
    kwgs: dict
        the extensible attribute

    Methods
    -------
    receiver_location:
        return the sample location of the receiver
    source_locations:
        return the location of the transmitter
    """

    def __init__(self, spacing, height, SNR, x_min, x_max, y_min, y_max,
                 collection_direction, **kwargs):
        self.spacing = spacing
        self.height = height
        self.SNR = SNR
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.collection_direction = collection_direction
        for key, val in kwargs.items():
            setattr(self, key, val)

    @property
    def receiver_location(self):
        """define and get the sample location of the receiver

        Returns
        -------
        res: 2-D-ndarry
            the location of the receiver, conclude the location of xyz axis
            every line is a location [x, y, z], the number of row represent
            the number of survey point
        """
        acq_area_xmin, acq_area_xmax = self.x_min, self.x_max
        acq_area_ymin, acq_area_ymax = self.y_min, self.y_max
        acquisition_spacing = self.spacing
        Nx = int((acq_area_xmax - acq_area_xmin) / acquisition_spacing + 1)
        Ny = int((acq_area_ymax - acq_area_ymin) / acquisition_spacing + 1)

        # Define receiver locations
        xrx, yrx, zrx = np.meshgrid(np.linspace(acq_area_xmin, acq_area_xmax, Nx),
                                    np.linspace(acq_area_ymin, acq_area_ymax, Ny),
                                    [self.height])
        receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]
        return receiver_locations

    @property
    def source_locations(self):
        """define and get the sample location of the transmitter

        Returns
        -------
        res: 2-D-ndarry
            the location of the receiver, conclude the location of xyz axis
            every line is a location [x, y, z], the number of row represent
            the number of survey point
        """
        # Defining transmitter locations
        acquisition_spacing = self.spacing
        acq_area_xmin, acq_area_xmax = self.x_min, self.x_max
        acq_area_ymin, acq_area_ymax = self.y_min, self.y_max
        Nx = int((acq_area_xmax - acq_area_xmin) / acquisition_spacing + 1)
        Ny = int((acq_area_ymax - acq_area_ymin) / acquisition_spacing + 1)
        xtx, ytx, ztx = np.meshgrid(np.linspace(acq_area_xmin, acq_area_xmax, Nx),
                                    np.linspace(acq_area_ymin, acq_area_ymax, Ny),
                                    [self.height])
        source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
        return source_locations
