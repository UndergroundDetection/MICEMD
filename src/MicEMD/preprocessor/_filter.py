import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


def s_filter(data, **kwargs):
    """
    对磁场响应数据进行滤波
    Parameters
    ----------
    data

    Returns
    -------

    """
    train = data[0]
    test = data[1]

    train_feature_lable = np.array(train)
    test_feature_lable = np.array(test)

    # 获取特征
    train_set = train_feature_lable[:, 0:-1]
    test_set = test_feature_lable[:, 0:-1]
    M1_x1_train_filtered = gaussian_filter1d(train_set, sigma=kwargs['sigma'])
    M2_x2_train_filtered = gaussian_filter1d(test_set, sigma=kwargs['sigma'])


    # 合并特征和标签
    train_filter = np.hstack((M1_x1_train_filtered, train_feature_lable[:, -1].reshape((-1, 1))))
    test_filter = np.hstack((M2_x2_train_filtered, test_feature_lable[:, -1].reshape((-1, 1))))

    return train_filter, test_filter