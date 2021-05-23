# -*- coding: utf-8 -*-
"""
The survey class, represent the survey in TDEM

Class:
- Survey: the implement class of the BaseTDEMSurvey
"""

__all__ = ['Survey']

from MicEMD import tdem
from abc import ABCMeta
from abc import abstractmethod


class BaseTDEMSurvey(metaclass=ABCMeta):
    """the abstract class about the survey in TDEM

    Attributes
    ----------
    Source: class
        the source in TDEM

    Methods:
    survey
        Returns the survey of the TDEM
    """
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
        _survey = tdem.Survey(self.source.source_list)
        return _survey
