# -*- coding: utf-8 -*-
"""
prepare the data

Methods:
- data_prepare: the interface of processing the data by Scalar
"""
__all__ = ['data_prepare']

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler


def data_prepare(response, task):
    """preparation of the tdem data

    Parameters
    ----------
    response: ndarry
        the data of response in TDEM
    task: str
        the specific task of classification

    Returns
    -------
    res: tuple
        conclude the train set and the test set.
    """

    data = response
    feature_lable = np.array(data)
    feature = feature_lable[:, 2:feature_lable.shape[1]]

    if task == 'material':
        # 优化目标 Y  向量化
        material_label = feature_lable[:, 0]

        X = feature

        # print("开始划分数据集")
        X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, material_label, test_size=0.3, random_state=5)

        # 归一化处理
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        X1_train = min_max_scaler.fit_transform(X1_train)
        X1_test = min_max_scaler.transform(X1_test)

        testing_set_material = np.hstack((X1_test, Y1_test.reshape((-1, 1))))
        training_set_material = np.hstack((X1_train, Y1_train.reshape((-1, 1))))

        return training_set_material, testing_set_material
    if task == 'shape':

        shape_label = feature_lable[:, 1]
        X = feature

        X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, shape_label, test_size=0.3, random_state=5)

        # 归一化处理
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))

        X2_train = min_max_scaler.fit_transform(X2_train)
        X2_test = min_max_scaler.transform(X2_test)

        testing_set_shape = np.hstack((X2_test, Y2_test.reshape((-1, 1))))
        training_set_shape = np.hstack((X2_train, Y2_train.reshape((-1, 1))))

        return training_set_shape, testing_set_shape
