from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.model_selection import cross_val_score
import math
from hyperopt import fmin, tpe, hp, partial, fmin
import os, random
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

MAX_EVAL = 200


def load_cls(x_test, algo, positive, negative, snr=20):
    est_path = f'./dataset/{negative}_{positive}_{snr}_estimator'
    os.makedirs(est_path, exist_ok=True)
    gcv = joblib.load(f'{est_path}/{algo}_{negative}_{positive}_{snr}.pickle')
    y_hat = gcv.predict(x_test)
    return y_hat


def contingency_table(y_pred1, y_pred2):
    """计算两个分类器的列联表
    Parameters
    ----------
    y_pred1：list 第一个分类器的预测结果
    y_pred2：list 第二个分类器的预测结果
    Returns：list 列联表
    -------

    """
    dct = {'00': 0, '01': 0, '10': 0, '11': 0}
    for i in range(len(y_pred1)):
        if y_pred1[i] == 0:
            if y_pred2[i] == y_pred1[i]:
                dct['00'] += 1
            else:
                dct['01'] += 1
        else:
            if y_pred2[i] == y_pred1[i]:
                dct['11'] = dct.get('11', 0) + 1
            else:
                dct['10'] = dct.get('10', 0) + 1
    return [dct['00'], dct['01'], dct['10'], dct['11']]


def get_dis(t):
    """根据列联表，计算df值"""
    dis = (t[1] + t[2]) / sum(t)
    return dis


def get_q(t):
    """计算q统计量"""
    q = (t[0] * t[3] - t[1] * t[2]) / (t[0] * t[3] + t[1] * t[2])
    return q


def get_cor(t):
    """计算相关系数"""
    cor = (t[0] * t[3] - t[1] * t[2]) / math.sqrt((t[0] + t[1]) * (t[0] + t[1]) * (t[2] + t[3]) * (t[1] + t[3]))
    return cor


def get_k(t):
    p1 = (t[0] + t[3]) / sum(t)
    p2 = ((t[0] + t[1]) * (t[0] + t[2]) + (t[2] + t[3]) * (t[2] + t[3])) / (sum(t) ** 2)
    k = (p1 - p2) / (1 - p2) if p2 != 1 else 0
    return k


def get_df(t):
    """根据列联表，计算df值"""
    df = t[0] / sum(t)
    return df


def df_system(cls_list, x_train, y_train, x_test, y_test, positive, negative, snr):
    """根据学习器list获取所有学习器之间的df值"""
    length = len(cls_list)
    y_pred_list = []
    for cls in cls_list:
        y_pred_list.append(
            load_cls(x_test, cls, positive, negative, snr))

    df = [[0 for _ in range(length + 1)] for _ in range(length)]
    for i in range(length):
        for j in range(i + 1, length):
            df[i][j] = df[j][i] = get_df(contingency_table(y_pred_list[i], y_pred_list[j]))
        df[i][-1] = sum(df[i]) / length
    return df


def dis_system(cls_list, x_train, y_train, x_test, y_test, positive, negative, snr):
    """根据学习器list获取所有学习器之间的不一致性度量值"""
    length = len(cls_list)
    y_pred_list = []
    for cls in cls_list:
        y_pred_list.append(
            load_cls(x_test, cls, positive, negative, snr))

    dis = [[0 for _ in range(length + 1)] for _ in range(length)]
    for i in range(length):
        for j in range(i + 1, length):
            dis[i][j] = dis[j][i] = get_dis(contingency_table(y_pred_list[i], y_pred_list[j]))
        dis[i][-1] = sum(dis[i]) / length
    return dis


def q_system(cls_list, x_train, y_train, x_test, y_test, positive, negative, snr):
    """根据学习器list获取所有学习器之间的q统计值"""
    length = len(cls_list)
    y_pred_list = []
    for cls in cls_list:
        y_pred_list.append(
            load_cls(x_test, cls, positive, negative, snr))

    q = [[0 for _ in range(length + 1)] for _ in range(length)]
    for i in range(length):
        for j in range(i + 1, length):
            q[i][j] = q[j][i] = get_q(contingency_table(y_pred_list[i], y_pred_list[j]))
        q[i][-1] = sum(q[i]) / length
    return q


def cor_system(cls_list, x_train, y_train, x_test, y_test, positive, negative, snr):
    """根据学习器list获取所有学习器之间的相关系数"""
    length = len(cls_list)
    y_pred_list = []
    for cls in cls_list:
        y_pred_list.append(
            load_cls(x_test, cls, positive, negative, snr))

    cor = [[0 for _ in range(length + 1)] for _ in range(length)]
    for i in range(length):
        for j in range(i + 1, length):
            cor[i][j] = cor[i][j] = get_cor(contingency_table(y_pred_list[i], y_pred_list[j]))
        cor[i][-1] = sum(cor[i]) / length
    return cor


def k_system(cls_list, x_train, y_train, x_test, y_test, positive, negative, snr):
    """根据学习器list获取所有学习器之间的k统计值"""
    length = len(cls_list)
    y_pred_list = []
    for cls in cls_list:
        y_pred_list.append(
            load_cls(x_test, cls, positive, negative, snr))

    k = [[0 for _ in range(length + 1)] for _ in range(length)]
    for i in range(length):
        for j in range(i + 1, length):
            k[i][j] = k[j][i] = get_k(contingency_table(y_pred_list[i], y_pred_list[j]))
        k[i][-1] = sum(k[i]) / length
    return k


def cls_filter_acc(cls_list, x, y, positive, negative, snr):
    """筛选学习率低于50%的学习器"""
    # 设置字典，使其对应的值为0
    c = {}
    for cls in cls_list:
        c[cls] = get_acc(x, y, cls, positive, negative, snr)
    return c


def cls_filter(cls_list, diversity_list, rule):
    c = Counter(cls_list)
    for k in c:
        c[k] -= 1
    for diversity, rule in zip(diversity_list, rule):
        diversity_sort = sorted(diversity, reverse=rule)
        for cls, idx in zip(cls_list, diversity):
            c[cls] += (diversity_sort.index(idx) + 1)
    return c


def get_res(x, y, cls, positive, negative, snr):
    """得到分类的效果，g_mean及auc指标"""
    est_path = f'./dataset/{negative}_{positive}_{snr}_estimator'
    os.makedirs(est_path, exist_ok=True)
    gcv = joblib.load(f'{est_path}/{cls}_{negative}_{positive}_{snr}.pickle')
    y_hat = gcv.predict(x)
    y_hat2 = gcv.predict_proba(x)[:, 1]
    auc_val = roc_auc_score(y, y_hat2)
    acc = accuracy_score(y, y_hat)
    c = confusion_matrix(y, y_hat)
    precision = c[0][0] / (c[0][0] + c[1][0]) if c[0][0] != 0 else 0  # 查准率,分子为零则结果为0
    recall = c[0][0] / (c[0][0] + c[0][1]) if c[0][0] != 0 else 0  # 查全率
    g_mean = np.sqrt(c[0][0] / (c[0][0] + c[0][1]) * (c[1][1] / (c[1][0] + c[1][1])))
    f1 = 2 * precision * recall / (precision + recall) if c[0][0] != 0 else 0
    beta = 2
    fb = ((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall) if c[0][0] != 0 else 0
    con_mat = c.flatten()
    print(cls, acc, g_mean, fb, con_mat, recall)
    return [g_mean]
    # return [acc]


def get_acc(x, y, cls, positive, negative, snr):
    """得到分类的效果，g_mean及auc指标"""
    est_path = f'./dataset/{negative}_{positive}_{snr}_estimator'
    os.makedirs(est_path, exist_ok=True)
    gcv = joblib.load(f'{est_path}/{cls}_{negative}_{positive}_{snr}.pickle')
    y_hat = gcv.predict(x)
    acc = accuracy_score(y, y_hat)
    c = confusion_matrix(y, y_hat)
    precision = c[0][0] / (c[0][0] + c[1][0]) if c[0][0] != 0 else 0  # 查准率,分子为零则结果为0
    recall = c[0][0] / (c[0][0] + c[0][1]) if c[0][0] != 0 else 0  # 查全率
    g_mean = np.sqrt(c[0][0] / (c[0][0] + c[0][1]) * (c[1][1] / (c[1][0] + c[1][1])))
    f1 = 2 * precision * recall / (precision + recall) if c[0][0] != 0 else 0
    beta = 2
    fb = ((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall) if c[0][0] != 0 else 0
    con_mat = c.flatten()
    print(cls, acc, g_mean, fb, con_mat, recall)
    return acc, recall


def get_weight(cls_list, x, y, positive, negative, snr, res=None):
    """获取学习器的权重"""

    weight_list = []
    for cls in cls_list:
        weight_list.append(
            get_res(x, y, cls, positive, negative, snr))
    g_auc = [sum(i) / len(i) for i in weight_list]

    weights = [i / sum(g_auc) for i in g_auc]

    return weights


def get_preds(cls_list, x, positive, negative, snr):
    y_pred_list = []
    for cls in cls_list:
        y_pred_list.append(
            load_cls(x, cls, positive, negative, snr))
    return y_pred_list


def hard_vote(y_pred_list, weights):
    """硬投票,返回集成后的预测结果"""
    vote0 = [0] * len(y_pred_list[0])
    vote1 = [0] * len(y_pred_list[0])
    votes = []
    for y_pred, weight in zip(y_pred_list, weights):
        for i, p in enumerate(y_pred):
            if p == 0:
                vote0[i] += weight
            else:
                vote1[i] += weight
    for v0, v1 in zip(vote0, vote1):
        # print(v0, v1)
        if v0 >= v1:
            votes.append(0)
        else:
            votes.append(1)
    return votes


def model_opt(x_train, y_train, x_test, y_test, algo, positive, negative, snr=20, res=None):
    """利用hyperopt来超参数寻优"""
    est_path = f'./dataset/{negative}_{positive}_{snr}_estimator'
    os.makedirs(est_path, exist_ok=True)
    gcv = None

    def factory(argsDict):
        if algo in ['mlp', 'mlp_reb', 'mlp_reb_pca']:
            cls = MLPClassifier()
        if algo in ['lg', 'lg_reb', 'lg_reb_pca']:
            cls = LogisticRegression()
        if algo in ['knn', 'knn_reb', 'knn_reb_pca']:
            cls = KNeighborsClassifier()
        if algo in ['dt', 'dt_reb', 'dt_reb_pca']:
            cls = DecisionTreeClassifier()
        if algo in ['svm', 'svm_reb', 'svm_reb_pca']:
            cls = SVC()
        if algo in ['nb', 'nb_reb', 'nb_reb_pca']:
            cls = GaussianNB()
        if algo in ['gdbt', 'gdbt_reb', 'gdbt_reb_pca']:
            cls = GradientBoostingClassifier()
        if algo in ['adaboost', 'adaboost_reb', 'adaboost_reb_pca']:
            cls = AdaBoostClassifier()
        if algo in ['randomForest', 'randomForest_reb', 'randomForest_reb_pca']:
            cls = RandomForestClassifier()
        return -np.mean(
            cross_val_score(estimator=cls, X=x_train, y=y_train.flatten()))  # scoring=make_scorer(metrics.recall_score)

    if algo in ['mlp', 'mlp_reb', 'mlp_reb_pca']:
        gcv = MLPClassifier(hidden_layer_sizes=(50,), activation='tanh', random_state=42)

    if algo in ['lg', 'lg_reb', 'lg_reb_pca']:
        gcv = LogisticRegression(solver='liblinear', random_state=42)

    if algo in ['knn', 'knn_reb', 'knn_reb_pca']:
        gcv = KNeighborsClassifier(5)

    if algo in ['dt', 'dt_reb', 'dt_reb_pca']:
        criterion = ['gini', 'entropy']
        space = {'criterion': hp.choice('criterion', criterion)}
        argsDict = fmin(factory, space, algo=partial(tpe.suggest, n_startup_jobs=20), max_evals=MAX_EVAL,
                        pass_expr_memo_ctrl=None)
        gcv = DecisionTreeClassifier(criterion=criterion[argsDict['criterion']], random_state=42)
    if algo in ['svm', 'svm_reb', 'svm_reb_pca']:
        gcv = SVC(kernel='linear', probability=True, random_state=42)
    if algo in ['nb', 'nb_reb', 'nb_reb_pca']:
        gcv = GaussianNB()
    if algo in ['adaboost', 'adaboost_reb', 'adaboost_reb_pca']:
        gcv = AdaBoostClassifier(random_state=42)
    if algo in ['gdbt', 'gdbt_reb', 'gdbt_reb_pca']:
        gcv = GradientBoostingClassifier(random_state=42)
    if algo in ['randomforest', 'randomforest_reb', 'randomforest_reb_pca']:
        gcv = RandomForestClassifier(random_state=42)

    gcv.fit(x_train, y_train.flatten())

    # 计算测试集的分类效果指标并保存，保存分类预测结果
    y_hat = gcv.predict(x_test)
    y_pred = y_hat[:, np.newaxis]
    res_y = np.hstack((y_test, y_pred))
    res_y = pd.DataFrame(res_y, columns=['y_true', 'y_pred'])
    res_y.to_csv(f'{est_path}/{algo}_{negative}_{positive}_{snr}_res.csv')  # 保存预测结果

    y_hat2 = gcv.predict_proba(x_test)[:, 1]
    auc_val = roc_auc_score(y_test, y_hat2)
    acc = accuracy_score(y_test, y_hat)
    c = confusion_matrix(y_test, y_hat)
    precision = c[0][0] / (c[0][0] + c[1][0]) if c[0][0] != 0 else 0  # 查准率,分子为零则结果为0
    recall = c[0][0] / (c[0][0] + c[0][1]) if c[0][0] != 0 else 0  # 查全率
    g_mean = np.sqrt(c[0][0] / (c[0][0] + c[0][1]) * (c[1][1] / (c[1][0] + c[1][1])))
    f1 = 2 * precision * recall / (precision + recall) if c[0][0] != 0 else 0
    beta = 2
    f1 = ((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall) if c[0][0] != 0 else 0
    con_mat = c.flatten()

    y_hat_train = gcv.predict(x_train)
    acc_train = accuracy_score(y_train, y_hat_train)

    with open(f'{est_path}/{algo}_{negative}_{positive}_{snr}_parameters.txt', 'w+') as file:
        file.write(str(gcv) + '\n')
        file.write(
            f'acc:{acc} precision:{precision} recall:{recall} f1:{f1} auc_val:{auc_val} g_mean:{g_mean} c:{con_mat} acc_t:{acc_train}')
    with open(f'{est_path}/{algo}_{negative}_{positive}_{snr}.pickle', 'wb') as file:
        joblib.dump(gcv, f'{est_path}/{algo}_{negative}_{positive}_{snr}.pickle')
    res.append([algo, acc, precision, recall, f1, g_mean, con_mat, acc_train])


def imbalance_train(snr, positive, negative):
    train_set = pd.read_csv(f'./dataset/train_dataset_{negative}_{positive}_{snr}db.csv', sep=',', encoding='utf-8',
                            index_col=0).values
    test_set = pd.read_csv(f'./dataset/test_dataset_{negative}_{positive}_{snr}db.csv', sep=',', encoding='utf-8',
                           index_col=0).values
    y_test = test_set[:, -1][:, np.newaxis]
    x_test = test_set[:, :-1]
    # print(x_test.shape, y_test.shape)

    y_train = train_set[:, -1][:, np.newaxis]
    x_train = train_set[:, :-1]

    pca = PCA(n_components=30)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    # 多数类欠采样
    rus = RandomUnderSampler(sampling_strategy=1 / 2, random_state=0)
    # rus = RandomUnderSampler(random_state=0)  # 50:100的时候，不能只保留50，这样就平衡了
    x_train, y_train = rus.fit_resample(x_train, y_train)
    print(sorted(Counter(y_train).items()))

    # 少数类过采样2
    smo = SMOTE(sampling_strategy="minority", k_neighbors=2, random_state=42)
    x_train, y_train = smo.fit_resample(x_train, y_train.flatten())
    print(x_train.shape, y_train.shape)
    print(Counter(y_train.flatten()))

    res = []

    model_opt(x_train, y_train, x_test, y_test, 'mlp_reb_pca', positive=positive, negative=negative, snr=snr, res=res)
    model_opt(x_train, y_train, x_test, y_test, 'lg_reb_pca', positive=positive, negative=negative, snr=snr, res=res)
    model_opt(x_train, y_train, x_test, y_test, 'knn_reb_pca', positive=positive, negative=negative, snr=snr, res=res)
    model_opt(x_train, y_train, x_test, y_test, 'dt_reb_pca', positive=positive, negative=negative, snr=snr, res=res)
    model_opt(x_train, y_train, x_test, y_test, 'svm_reb_pca', positive=positive, negative=negative, snr=snr, res=res)
    model_opt(x_train, y_train, x_test, y_test, 'nb_reb_pca', positive=positive, negative=negative, snr=snr, res=res)
    model_opt(x_train, y_train, x_test, y_test, 'adaboost_reb_pca', positive=positive, negative=negative, snr=snr,
              res=res)
    model_opt(x_train, y_train, x_test, y_test, 'gdbt_reb_pca', positive=positive, negative=negative, snr=snr, res=res)
    model_opt(x_train, y_train, x_test, y_test, 'randomforest_reb_pca', positive=positive, negative=negative, snr=snr,
              res=res)
    est_path = f'./dataset/{negative}_{positive}_{snr}_estimator'
    res = pd.DataFrame(res, columns=['algo', 'acc', 'precision', 'recall', 'f1', 'g_mean', 'con_mat', 'acc_train'])
    res.to_csv(f'{est_path}/allCLs_reb_pca_res_{negative}_{positive}_{snr}.csv')
    res.to_csv(f'{est_path}/allCLs_reb_res_{negative}_{positive}_{snr}.csv')
    return x_train, x_test, y_train, y_test


def imbalance_ensemble(x_train, y_train, x_test, y_test, positive, negative, snr=20):
    cls_list = ['lg_reb_pca', 'dt_reb_pca', 'nb_reb_pca', 'knn_reb_pca', 'mlp_reb_pca', 'svm_reb_pca', 'gdbt_reb_pca',
                'adaboost_reb_pca', 'randomforest_reb_pca']

    # 剔除acc and recall低于50%的学习器
    acc_rank = cls_filter_acc(cls_list, x_test, y_test, positive=positive, negative=negative, snr=snr)
    print(acc_rank)
    for acc in acc_rank:
        if acc_rank[acc][0] <= 0.7 or acc_rank[acc][1] <= 0.5:
            cls_list.remove(acc)
    print('第一次筛选后', cls_list)

    # 计算多样性指标
    dis = dis_system(cls_list, x_train, y_train, x_test, y_test, positive=positive, negative=negative, snr=snr)
    dis = pd.DataFrame(dis, columns=cls_list + ['avg'], index=cls_list)
    # print(dis)
    k = k_system(cls_list, x_train, y_train, x_test, y_test, positive=positive, negative=negative, snr=snr)
    k = pd.DataFrame(k, columns=cls_list + ['avg'], index=cls_list)
    cor = q_system(cls_list, x_train, y_train, x_test, y_test, positive=positive, negative=negative, snr=snr)
    print(cor)
    cor = pd.DataFrame(cor, columns=cls_list + ['avg'], index=cls_list)

    # 组合多样性指标
    d = [dis['avg'].values, k['avg'].values, cor['avg'].values]
    # 筛选多样性差的学习器，降低模型复杂度
    rule = [True, False, False]
    rank = cls_filter(cls_list, d, rule)
    print(rank)
    # 剔除多样性最低的学习器
    name_cls = rank.most_common(1)[0][0]
    print(name_cls)
    # 判断学习器个数，再筛选
    if len(cls_list) > 3:
        cls_list.remove(name_cls)
    print(cls_list)
    # cls_list.remove(name_cls)

    # 集成
    weights = get_weight(cls_list, x_train, y_train, positive=positive, negative=negative, snr=snr, res=None)
    print('训练集', weights, sum(weights))
    wei = get_weight(cls_list, x_test, y_test, positive=positive, negative=negative, snr=snr, res=None)
    print('测试集', wei)

    y_preds = get_preds(cls_list, x_test, positive=positive, negative=negative, snr=snr)
    y_hat = hard_vote(y_preds, weights)
    # y_hat = avg_vote(y_preds)

    acc = accuracy_score(y_test.flatten(), y_hat)
    c = confusion_matrix(y_test.flatten(), y_hat)
    precision = c[0][0] / (c[0][0] + c[1][0]) if c[0][0] != 0 else 0  # 查准率,分子为零则结果为0
    recall = c[0][0] / (c[0][0] + c[0][1]) if c[0][0] != 0 else 0  # 查全率
    g_mean = np.sqrt(c[0][0] / (c[0][0] + c[0][1]) * (c[1][1] / (c[1][0] + c[1][1])))
    f1 = 2 * precision * recall / (precision + recall) if c[0][0] != 0 else 0
    beta = 2
    fb = ((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall) if c[0][0] != 0 else 0
    con_mat = c.flatten()

    print(acc, precision, recall, fb, g_mean, con_mat)


def imbalance_learn(positive, negative, snr=20):
    x_train, x_test, y_train, y_test = imbalance_train(snr, positive, negative)
    imbalance_ensemble(x_train, y_train, x_test, y_test, positive, negative, snr=20)
