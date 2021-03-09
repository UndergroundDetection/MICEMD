__all__ = ['ForwardResult', 'InvResult']

from abc import ABCMeta
from abc import abstractmethod


class BaseForwardResult(metaclass=ABCMeta):
    pass


class BaseClsResult(metaclass=ABCMeta):
    pass


class ForwardResult(BaseForwardResult):
    def __init__(self, result, **kwargs):
        pass


class ClsResult(BaseClsResult):
    def __init__(self, result, **kwargs):
        pass
