# -*- coding: utf-8 -*-
"""
The ANN classification method in TDEM

Methods:
    MLP: the ANN classification method
"""
__all__ = ['MLP']

from sklearn.neural_network import MLPClassifier  # 'tanh'，lbfgs,(50,)
from sklearn.metrics import accuracy_score
import numpy as np


def MLP(train_set, test_set, solver='lbfgs', hidden_layer_sizes=(50,), activation='tanh'):
    """the ANN classification algorithm

    Parameters
    ----------
    train_set: ndarry
        the train set
    test_set: ndarry
        the test set
    solver: str
        the optimization of the ANN
    hidden_layer_sizes: tuple
        the ANN hidden layer sizes
    activation: str
        the activation of the ANN

    Returns
    -------
    res: tuple
        (accuracy, y_pred, y_true),the accuracy of classification, the predict value and
        the true value

    """
    res = {}
    clf = MLPClassifier(solver=solver, hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=10000)
    trainTarget = np.array(train_set)[:, -1]
    testTarget = np.array(test_set)[:, -1]

    tmp_clf = clf.fit(train_set, trainTarget)

    tmp_clf.predict(test_set)

    y_pred = (clf.fit(train_set[:, :-1], trainTarget)).predict(test_set[:, :-1])
    y_true = testTarget
    tmp_accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

    res = {'accuracy': tmp_accuracy, 'y_pred': y_pred, 'y_true': y_true}
    return res
