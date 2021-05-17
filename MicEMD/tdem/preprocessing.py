__all__ = ['preprocessing']

from MicEMD.handler import TDEMHandler
from MicEMD.preprocessor import *
from MicEMD.tdem import *


def preprocessing(ForwardResult, task, dim_red_method=None, save=True):
    original_data = data_prepare(ForwardResult, task)
    handler = TDEMHandler(ForwardResult, None)
    handler.save_preparation(original_data, task, save)
    if dim_red_method == 'SF':
        dim_reduction_data = statistic_feature(original_data)
        preResult = PreResult(dim_reduction_data, task, dim_red_method, original_data, ForwardResult)
        handler = TDEMHandler(ForwardResult, preResult)
        handler.save_dim_reduction(save)
    else:
        preResult = PreResult(None, task, dim_red_method, original_data, ForwardResult)
    return preResult
