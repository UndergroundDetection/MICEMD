__all__ = ['Source']
import SimPEG.electromagnetics.frequency_domain as fdem
from abc import ABCMeta
from abc import abstractmethod


class BaseSource(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, Target, Detector, Collection, *args):
        self.target = Target
        self.detector = Detector
        self.collection = Collection

    @abstractmethod
    def source_list(self):
        pass


class Source(BaseSource):

    def __init__(self, Target, Detector, Collection, *args):
        BaseSource.__init__(self, Target, Detector, Collection)

    @property
    def source_list(self):
        pass





