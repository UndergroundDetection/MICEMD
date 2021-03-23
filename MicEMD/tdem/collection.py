# -*- coding: utf-8 -*-
__all__ = ['Collection']


import numpy as np
from ..utils import mkvc


class Collection(object):
    """

    Attributes
    ----------
    t_split: int
        the sampled times of the response every second
    SNR: int
        the Signal to Noise Ratio

    """

    def __init__(self, t_split, SNR, **kwargs):
        self.t_split = t_split
        self.SNR = SNR



