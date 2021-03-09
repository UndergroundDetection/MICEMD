# -*- coding: utf-8 -*-
__all__ = ['Collection']


import numpy as np
from ..utils import mkvc


class Collection(object):
    """

    Attributes
    ----------

    """

    def __init__(self, spacing, height, SNR, x_min, x_max, y_min, y_max,
                 collection_direction):
        self.spacing = spacing
        self.height = height
        self.SNR = SNR
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.collection_direction = collection_direction

    @property
    def receiver_location(self):
        """get the sample location

        Returns
        -------

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
