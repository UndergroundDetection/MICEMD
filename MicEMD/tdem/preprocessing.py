# -*- coding: utf-8 -*-
"""
preprocess the response data in TDEM


Methods:
- preprocess: the interface of the preprocessor in TDEM
"""
__all__ = ['preprocess']

from MicEMD.handler import TDEMHandler
from MicEMD.preprocessor import *
from MicEMD.tdem import *


def preprocess(response, dim_red_method=None):
    """The data were normalized and dimensionalized

    Parameters
    ----------
    response: ndarry
        the received response

    dim_red_method: str
        the method of the dimensionality reduction

    Returns
    -------
    res: tuple
        the train set and test set of after dimensionality reduction
        for example:
        res[0] represent the train set after dimensionality reduction
        res[1] represent the test set after dimensionality reduction

    """

    original_data = response
    dim_reduction_data = None
    if dim_red_method == 'SF':
        dim_reduction_data = statistic_feature(original_data)

    res = dim_reduction_data
    return res
