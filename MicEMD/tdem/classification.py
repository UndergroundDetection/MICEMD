# -*- coding: utf-8 -*-
__all__ = ['Classification', 'classify']

from abc import ABCMeta
from abc import abstractmethod



class BaseClassification(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, ForwardResult):
        pass

    @abstractmethod
    def true_properties(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def error(self):
        pass

    @abstractmethod
    def inv_objective_function(self):
        pass


class Classification(BaseClassification):

    def __init__(self, ForwardResult, *args, **kwargs):
        pass

    @property
    def true_properties(self):
        pass

    def inv_objective_function(self):
        pass

    def run(self):
        pass

    @property
    def error(self):
        pass


def classify(method, ForwardResult, save, *args, **kwargs):
    pass
