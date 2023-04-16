# -*- coding: utf-8 -*-
"""
The classification class in TDEM

Class:
- classification: the implement class of the BaseTDEMSimulation

Methods:
- simulate: the interface of the simulation in TDEM
"""
__all__ = ['Classification', 'classify']

from abc import ABCMeta
from abc import abstractmethod

from ..classification.classfiy import classify_method


class BaseTDEMClassification(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, data, method, cls_para):
        self.data = data
        self.method = method
        self.cls_para = cls_para

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def error(self):
        pass

class Classification(BaseTDEMClassification):
    """the class of the classification algorithm

    Attributes
    ----------
    data: tuple
        conclude the train set and the test set
    method: str
        the method of classification
    cls_para: dict
        the Parameters for the classification algorithm

    Methods:
    -------
    - run: run the the classification algorithm and return the result
    - error: return the res of the classification
    """

    def __init__(self, data, method, cls_para, *args, **kwargs):
        self.data = data
        self.method = method
        self.cls_para = cls_para

    def run(self):
        self.res = classify_method(self.data[0], self.data[1], self.method, self.cls_para)

    @property
    def error(self):
        return self.res


def classify(data_set, cls_method, cls_para=None, *args, **kwargs):
    """the interface of the classification of the target

    Parameters
    ----------
    data_set: tuple
        conclude the train set and the test set
    cls_method: str
        the method of classification
    cls_para: dict
        the Parameters for the classification algorithm

    Returns
    -------
    res: dict
        keys: ['accuracy', 'y_pred', 'y_true'], represent the accuracy ,
        predict value and true value of the classification

    """
    if cls_method in ['人工神经网络', 'ANN']:
        cls_method = 'ANN'
    _classify = Classification(data_set, cls_method, cls_para)
    _classify.run()
    res = _classify.error
    return res
