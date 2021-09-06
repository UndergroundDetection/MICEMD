# -*- coding: utf-8 -*-
"""
The survey class, represent the survey in FDEM

Class:
- Survey: the implement class of the BaseFDEMSurvey
"""
__all__ = ['Survey']

import SimPEG.electromagnetics.frequency_domain as fdem
from abc import ABCMeta
from abc import abstractmethod


class BaseFDEMSurvey(metaclass=ABCMeta):
    """the abstract class about the survey in FDEM

    Attributes
    ----------
    Source: class
        the source in FDEM

    Methods:
    survey
        Returns the survey of the FDEM
    """

    @abstractmethod
    def __init__(self, Source):
        self.source = Source

    @abstractmethod
    def survey(self):
        _survey = fdem.Survey(self.source.source_list)
        return _survey


class Survey(BaseFDEMSurvey):

    def __init__(self, Source):
        BaseFDEMSurvey.__init__(self, Source)

    @property
    def survey(self):
        _survey = fdem.Survey(self.source.source_list)
        return _survey
