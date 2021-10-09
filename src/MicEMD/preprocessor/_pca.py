__all__ = ['pca']
import numpy as np
from sklearn.decomposition import PCA


def pca(data, *, n_components):
    """the PCA dimensionality reduction algorithms

    Parameters
    ----------
    data： tuple
        conclude the train set and test set
    n_components： int
        the parameter of the pca

    Returns
    -------
        return the train set and test set after dimensionality
    """

    train = data[0]
    test = data[1]

    train_feature_lable = np.array(train)
    test_feature_lable = np.array(test)

    train_set = train_feature_lable[:, 0:-1]
    test_set = test_feature_lable[:, 0:-1]

    pca_model = PCA(n_components=n_components).fit(train_set)  # 对离差标准化的训练集，生成规则
    # 应用规则到训练集
    train_pca = pca_model.transform(train_set)


    # 应用规则到测试集
    test_pca = pca_model.transform(test_set)

    train_pca = np.hstack((train_pca, train_feature_lable[:, -1].reshape((-1, 1))))
    test_pca = np.hstack((test_pca, test_feature_lable[:, -1].reshape((-1, 1))))

    return train_pca, test_pca
