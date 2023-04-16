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


class BaseSimulation(metaclass=ABCMeta):
    """The abstract Simulation base class

    Parameters
    ---------
    model: class
        the model class
    Methods
    -------
    pred:
        Returns the observed data
    """

    @abstractmethod
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def pred(self):
        pass


class Simulation(BaseSimulation):
    """simulate the electromagnetic response

    Parameters
    ----------
    model: class
        the model class which call the pred method to generate the observed data

    Methods
    -------
    pred: ndarry
        Returns the observed data

    """

    def __init__(self, model):
        BaseSimulation.__init__(self, model)

    def pred(self):
        result = self.model.dpred()
        self.result = result
        return result


def simulate(target, detector, collection, model='simpeg', *args, **kwargs):
    """the simulate interface is used to handle organization and dispatch of directives of the simulation

    Parameters
    ----------
    target: class
        the target class
    detector: class
        the detector class
    collection: class
        the collection class
    model: class
        the model class

    Returns
    -------
    result: ndarry
        the observed data conclude the position and the magnetic field intensity

    """
    if model == 'simpeg':
        source = Source(target, detector, collection)
        survey = Survey(source)
        _model = Model(survey)
        simulation = Simulation(_model)
        result = simulation.pred()
        return result
    elif model == 'dipole':
        source = Source(target, detector, collection)
        survey = Survey(source)
        _model = DipoleModle(survey)
        simulation = Simulation(_model)
        result = simulation.pred()
        return result
