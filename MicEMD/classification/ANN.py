__all__=['MLP']
from sklearn.neural_network import MLPClassifier  # 'tanh'，lbfgs,(50,)

from sklearn.metrics import accuracy_score
import os
import numpy as np
import pandas as pd
import time


def MLP(train_set, test_set, solver='lbfgs', hidden_layer_sizes=(50,), activation='tanh'):
    clf = MLPClassifier(solver=solver, hidden_layer_sizes=hidden_layer_sizes, activation=activation)
    trainTarget = np.array(train_set)[:, -1]
    testTarget = np.array(test_set)[:, -1]

    tmp_clf = clf.fit(train_set, trainTarget)

    tmp_clf.predict(test_set)

    y_pred = (clf.fit(train_set[:, :-1], trainTarget)).predict(test_set[:, :-1])
    y_true = testTarget
    tmp_accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    return tmp_accuracy, y_pred, y_true
