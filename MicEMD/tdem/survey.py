import MicEMD.tdem
from abc import ABCMeta
from abc import abstractmethod

__all__ = ['Survey']

from MicEMD import tdem


class BaseTDEMSurvey(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, Source):
        self.source = Source

    @abstractmethod
    def survey(self):
        _survey = tdem.Survey(self.source.source_list)
        return _survey


class Survey(BaseTDEMSurvey):

    def __init__(self, Source):
        BaseTDEMSurvey.__init__(self, Source)

    @property
    def survey(self):
        pass

