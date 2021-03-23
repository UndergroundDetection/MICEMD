__all__ = ['Simulation', 'simulate']

import pandas as pd
from abc import ABCMeta
from abc import abstractmethod
from .survey import *
from .source import *
from .model import *
from .results import *
from ..handler import TDEMHandler
import matplotlib.pyplot as plt


class BaseTDEMSimulation(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def pred(self):
        pass


class Simulation(BaseTDEMSimulation):

    def __init__(self, model):
        BaseTDEMSimulation.__init__(self, model)

    def pred(self):
        result = self.model.dpred()
        return result


def simulate(target, detector, collection, model='dipole', save=True, show=False, *args, **kwargs):
    source = Source(target, detector, collection)
    survey = Survey(source)
    _model = Model(survey)
    simulation = Simulation(_model)
    result = simulation.pred()
    result = ForwardResult((result, simulation, {'method': 'dipole'}))
    handler = TDEMHandler(result, None)
    handler.save_forward(save)

    return result
