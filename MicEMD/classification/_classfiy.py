from .ANN import *


def classify_method(train_set, test_set, cls_method='ANN', cls_para=None):
    if cls_method == 'ANN':
        if cls_para is not None:
            solver = cls_para['solver']
            hidden_layer_sizes = cls_para['hidden_layer_sizes']
            activation = cls_para['activation']
            res = MLP(train_set, test_set, solver, hidden_layer_sizes, activation)
        else:
            res = MLP(train_set, test_set)
    return res
