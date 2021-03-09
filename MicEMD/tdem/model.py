__all__ = ['Model']
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
from ..utils import RotationMatrix
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG import maps
import SimPEG.electromagnetics.frequency_domain as fdem

from scipy.constants import mu_0




class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, Survey):
        self.survey = Survey

    @abstractmethod
    def dpred(self):
        pass


class Model(BaseModel):
    def __init__(self, Survey):
        BaseModel.__init__(self, Survey)

    def dpred(self):
        pass





