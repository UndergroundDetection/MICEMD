import SimPEG.electromagnetics.frequency_domain as fdem
from abc import ABCMeta
from abc import abstractmethod

__all__ = ['Survey']


class BaseSurvey(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, Source):
        self.source = Source

    @abstractmethod
    def survey(self):
        _survey = fdem.Survey(self.source.source_list)
        return _survey


class Survey(BaseSurvey):

    def __init__(self, Source):
        BaseSurvey.__init__(self, Source)

    @property
    def survey(self):
        _survey = fdem.Survey(self.source.source_list)
        return _survey
