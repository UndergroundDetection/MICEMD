__all__ = ['ForwardResult', 'InvResult']

from abc import ABCMeta
from abc import abstractmethod


class BaseForwardResult(metaclass=ABCMeta):
    pass


class BaseInvResult(metaclass=ABCMeta):
    pass


class ForwardResult(BaseForwardResult):
    def __init__(self, result, **kwargs):
        self.receiver_locations = result[0][0]
        self.mag_data = result[0][1]
        self.mesh = result[0][2]
        self.mapped_model = result[0][3]
        self.simulation = result[1]
        self.condition = result[2]


class InvResult(BaseInvResult):
    def __init__(self, result, **kwargs):
        self.estimate_parameters = result[0]
        self.true_parameters = result[1]
        self.error = result[2]
        self.condition = result[3]
