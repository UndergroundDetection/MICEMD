# -*- coding: utf-8 -*-
"""
The classification method in TDEM

Methods:
- classify_method: Provide the name of the classification method and its parameters
 to solve the classification problem
"""
__all__ = ['classify_method']
from .ANN import *
from .DT import *
from .KNN import *


def classify_method(train_set, test_set, cls_method='ANN', cls_para=None):
    """solve the classification problem of the target

    Parameters
    ----------
    train_set: ndarry
        the data of the train set
    test_set: ndarry
        the data of the test set
    cls_method: str
        the name of the classification method
    cls_para: dict
        the parameters of the classifier

    Returns
    -------
    res: tuple
        (accuracy, y_pred, y_true),the accuracy of classification, the predict value and
        the true value

    """
    res = None
    if cls_method == 'ANN':
        if cls_para is not None:
            solver = cls_para['solver']
            hidden_layer_sizes = cls_para['hidden_layer_sizes']
            activation = cls_para['activation']
            res = MLP(train_set, test_set, solver, hidden_layer_sizes, activation)
        else:
            res = MLP(train_set, test_set)
    elif cls_method == 'DT':
        res = DT(train_set, test_set)
    elif cls_method == 'KNN':
        res = KNN(train_set, test_set)
    return res
