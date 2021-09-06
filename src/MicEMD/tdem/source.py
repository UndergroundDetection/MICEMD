# -*- coding: utf-8 -*-
"""
The Source class, represent the source in TDEM

Class:
- Source: implement class of the BaseTDEMSource in TDEM
"""
__all__ = ['Source']

from abc import ABCMeta
from abc import abstractmethod


class BaseTDEMSource(metaclass=ABCMeta):
    """the abstract class about the source in TDEM

    Attributes
    ----------
    Target: class
        the target in TDEM
    Detector: class
        the detector class in TDEM
    Collection: class
        the Collection class in TDEM

    Methods:
    source_list
        Returns the source list of the TDEM
    """

    @abstractmethod
    def __init__(self, Target, Detector, Collection, *args):
        self.target = Target
        self.detector = Detector
        self.collection = Collection

    @abstractmethod
    def source_list(self):
        pass


class Source(BaseTDEMSource):

    def __init__(self, Target, Detector, Collection, *args):
        BaseTDEMSource.__init__(self, Target, Detector, Collection)

    @property
    def source_list(self):
        pass







