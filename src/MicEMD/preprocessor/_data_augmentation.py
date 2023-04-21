import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.decomposition import PCA


def data_aug_SMOTE(x_train, y_train, rus_rate, *args, **kwargs):

    # 多数类欠采样
    # rus = RandomUnderSampler(sampling_strategy=1/2, random_state=0)
    rus = RandomUnderSampler(sampling_strategy=rus_rate, random_state=0)
    # rus = RandomUnderSampler(random_state=0)  # 50:100的时候，不能只保留50，这样就平衡了
    x_train, y_train = rus.fit_resample(x_train, y_train)


    # 少数类过采样1
    # ros = RandomOverSampler(random_state=0)
    # x_train, y_train = ros.fit_resample(x_train, y_train)
    # print(sorted(Counter(y_train).items()))

    # 少数类过采样2
    smo = SMOTE(sampling_strategy="minority", k_neighbors=2, random_state=42)
    x_train, y_train = smo.fit_resample(x_train, y_train.flatten())
    print(x_train.shape, y_train.shape)
    return x_train, y_train
