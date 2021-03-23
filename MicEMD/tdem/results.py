__all__ = ['ForwardResult', 'ClsResult']

from abc import ABCMeta
from abc import abstractmethod


class BaseForwardResult(metaclass=ABCMeta):
    pass


class BaseClsResult(metaclass=ABCMeta):
    pass


class ForwardResult(BaseForwardResult):
    def __init__(self, result, **kwargs):
        self.response = result[0][0]
        self.sample = result[0][1]
        self.simulation = result[1]
        self.condition = result[2]


class ClsResult(BaseClsResult):
    def __init__(self, result, **kwargs):
        pass
