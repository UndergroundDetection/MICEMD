# -*- coding: utf-8 -*-
__all__ = ['Classification', 'classify']

from abc import ABCMeta
from abc import abstractmethod

from . import ClsResult
from ..classification._classfiy import classify_method
from ..handler import TDEMHandler
from ..preprocessor import *
from ..classification.ANN import *


class BaseClassification(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, PreResult):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def error(self):
        pass


class Classification(BaseClassification):

    def __init__(self, PreResult, method, cls_para, *args, **kwargs):
        self.preResult = PreResult
        self.method = method
        self.cls_para = cls_para

    def run(self):
        if self.method == 'ANN':
            self.res = classify_method(self.preResult.train_set, self.preResult.test_set, self.method, self.cls_para)
        else:
            self.res = classify_method(self.preResult.train_set, self.preResult.test_set, self.method, self.cls_para)

    @property
    def error(self):
        return self.res


def classify(preResult, cls_method, cls_para=None, save=True, show=True, *args, **kwargs):
    if cls_method in ['人工神经网络', 'ANN']:
        method = 'ANN'
    _classify = Classification(preResult, cls_method, cls_para)
    _classify.run()
    res = _classify.error
    clsResult = ClsResult(res, cls_method, cls_para)
    handler = TDEMHandler(preResult.forwardResult, preResult, clsResult)
    handler.plot_confusion_matrix(show)
    return clsResult
