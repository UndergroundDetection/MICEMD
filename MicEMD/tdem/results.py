# __all__ = ['ForwardResult', 'ClsResult', 'PreResult']
#
# from abc import ABCMeta
#
#
# class BaseForwardResult(metaclass=ABCMeta):
#     pass
#
#
# class BaseClsResult(metaclass=ABCMeta):
#     pass
#
#
# class ForwardResult(BaseForwardResult):
#     def __init__(self, result, **kwargs):
#         self.response = result[0][0]
#         self.sample = result[0][1]
#         self.simulation = result[1]
#         self.condition = result[2]
#
#
# class ClsResult(BaseClsResult):
#     def __init__(self, result, cls_method, cls_para,**kwargs):
#         self.cls_result = result
#         self.cls_method = cls_method
#         self.cls_para = cls_para
#
#
# class PreResult:
#     def __init__(self, data, task, dim_red_method, original_data, ForwardResult, **kwargs):
#         if dim_red_method is not None and data is not None:
#             self.train_set = data[0]
#             self.test_set = data[1]
#             self.task = task
#             self.dim_red_method = dim_red_method
#             self.train_set_original = original_data[0]
#             self.test_set_original = original_data[1]
#             self.forwardResult = ForwardResult
#         else:
#             self.train_set = original_data[0]
#             self.test_set = original_data[1]
#             self.task = task
#             self.dim_red_method = dim_red_method
#             self.forwardResult = ForwardResult
