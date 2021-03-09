__all__ = ['Simulation', 'simulate']

from abc import ABCMeta
from abc import abstractmethod
from .survey import *
from .source import *
from .model import *
from .results import *
from ..handler import Handler
import matplotlib.pyplot as plt


class BaseSimulation(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def pred(self):
        pass


class Simulation(BaseSimulation):

    def __init__(self, model):
        BaseSimulation.__init__(self, model)

    def pred(self):
        result = self.model.dpred()
        return result


def simulate(target, detector, collection, model='simpeg', save=True, show=False, *args, **kwargs):
    pass
