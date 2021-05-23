# -*- coding: utf-8 -*-
"""
the statistic feature method of the dimensionality reduction

Methods:
- statistic_feature: Reducing dimensions of the original data in TDEM
"""
__all__ = ['statistic_feature']
import numpy as np


def my_mad(X=None):
    return np.sum(np.absolute(X - np.mean(X, axis=1).reshape((-1, 1))), axis=1).reshape((-1, 1)) / X.shape[1]


def skeness(X=None):
    skeness = []
    for i in range(X.shape[0]):
        mean = np.mean(X[i, :])
        std = np.std(X[i, :])
        skeness.append((np.sum(np.power(X[i, :] - mean, 3)) / X.shape[1]) / np.power(std, 3))
    return np.array(skeness).reshape((-1, 1))


def statistic_feature(data):
    train = data[0]
    test = data[1]

    # 数据概况预览
    train_feature_lable = np.array(train)
    test_feature_lable = np.array(test)

    train_set = train_feature_lable[:, 0:400]
    test_set = test_feature_lable[:, 0:400]

    # compute the feature of the statistic
    train_statistic = np.max(train_set, axis=1).reshape((-1, 1))
    train_statistic = np.hstack((train_statistic, np.min(train_set, axis=1).reshape((-1, 1))))
    train_statistic = np.hstack((train_statistic, np.mean(train_set, axis=1).reshape((-1, 1))))
    train_statistic = np.hstack((train_statistic, np.var(train_set, axis=1).reshape((-1, 1))))
    train_statistic = np.hstack((train_statistic, np.std(train_set, axis=1).reshape((-1, 1))))
    train_statistic = np.hstack((train_statistic, my_mad(X=train_set)))
    train_statistic = np.hstack((train_statistic, skeness(X=train_set)))
    train_statistic = np.hstack((train_statistic, train_feature_lable[:, -1].reshape((-1, 1))))

    test_statistic = np.max(test_set, axis=1).reshape((-1, 1))
    test_statistic = np.hstack((test_statistic, np.min(test_set, axis=1).reshape((-1, 1))))
    test_statistic = np.hstack((test_statistic, np.mean(test_set, axis=1).reshape((-1, 1))))
    test_statistic = np.hstack((test_statistic, np.var(test_set, axis=1).reshape((-1, 1))))
    test_statistic = np.hstack((test_statistic, np.std(test_set, axis=1).reshape((-1, 1))))
    test_statistic = np.hstack((test_statistic, my_mad(X=test_set)))
    test_statistic = np.hstack((test_statistic, skeness(X=test_set)))
    test_statistic = np.hstack((test_statistic, test_feature_lable[:, -1].reshape((-1, 1))))

    return train_statistic, test_statistic

