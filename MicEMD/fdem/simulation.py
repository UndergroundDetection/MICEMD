# -*- coding: utf-8 -*-
"""
The simulation in TDEM

Class:
- Simulation: the implement class of the BaseFDEMSimulation

Methods:
- simulate: the interface of the simulation in FDEM
"""
__all__ = ['Simulation', 'simulate']

from abc import ABCMeta
from abc import abstractmethod
from .survey import *
from .source import *
from .model import *
from .results import *
from ..handler import FDEMHandler
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
        self.result = result
        return result


def simulate(target, detector, collection, model='simpeg', save=True, show=True, *args, **kwargs):
    if model == 'simpeg':
        source = Source(target, detector, collection)
        survey = Survey(source)
        _model = Model(survey)
        simulation = Simulation(_model)
        result = simulation.pred()
        # result = ForwardResult((result, simulation, {'method': model}))
        # handler = FDEMHandler(result, None)
        # handler.save_forward(save)
        # if show:
        #     fig1 = plt.figure()
        #     fig2 = plt.figure()
        #     fig3 = plt.figure()
        #     handler.show_fdem_detection_scenario(fig1)
        #     handler.show_fdem_mag_map(fig2)
        #     handler.show_discretize(fig3)

        return result
