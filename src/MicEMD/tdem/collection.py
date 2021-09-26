# -*- coding: utf-8 -*-
"""
The Collection class, conclude the parameters about collecting in underground detection system

Class:
- Collection: the class conclude the parameters about collecting in TDEM
"""
__all__ = ['Collection']


class Collection(object):
    """the class conclude the parameters about collecting

    Attributes
    ----------
    t_split: int
        the sampled times of the response every second
    SNR: int
        the Signal to Noise Ratio
    kwgs: dict
        the extensible attribute
    """

    def __init__(self, t_split, snr, **kwargs):
        self.t_split = t_split
        self.SNR = snr
        for key, val in kwargs.items():
            setattr(self, key, val)



