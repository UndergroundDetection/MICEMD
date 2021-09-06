# -*- coding: utf-8 -*-
"""
The simulation in TDEM

Class:
- Simulation: the implement class of the BaseTDEMSimulation

Methods:
- simulate: the interface of the simulation in TDEM
"""

__all__ = ['Simulation', 'simulate']

from abc import ABCMeta
from abc import abstractmethod
from .survey import *
from .source import *
from .model import *


class BaseTDEMSimulation(metaclass=ABCMeta):
    """the abstract class about the simulation in TDEM

    Attributes
    ----------
    model: class
    the model in TDEM

    Methods:
    pred
        Returns the forward_result of the TDEM
    """
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


def simulate(target, detector, collection, model='dipole', *args, **kwargs):
    """the interface of the simulation

    Parameters
    ----------
    target: class
        the target of the TDEM
    detector: class
        the detector of the TDEM
    collection: class
        the collection of the TDEM
    model: str
        the name of the model

    Returns
    -------
    res: tuple
        the result of the method dpred in model class
    """
    if model == 'dipole':
        source = Source(target, detector, collection)
        survey = Survey(source)
        _model = Model(survey)
        simulation = Simulation(_model)
        result = simulation.pred()
        return result
    else:
        pass
