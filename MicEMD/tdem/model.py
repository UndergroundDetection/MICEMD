__all__ = ['Model']

from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import pandas as pd
from functools import partial
import os
import matplotlib.pyplot as plt


class BaseTDEMModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, Survey):
        self.survey = Survey

    @abstractmethod
    def dpred(self):
        pass


class Model(BaseTDEMModel):
    def __init__(self, Survey):
        BaseTDEMModel.__init__(self, Survey)

    def parameter_sphere(self, c, c0, d, r):  # c是相对磁化率 μr；d是电导率 σ；r是半径；t是时间区间
        # pi = 3.14
        pi = np.pi
        e1 = pi
        e2 = pi * 1.5  # 二分范围
        # c=1.00125;              #相对磁化率 μr
        # c0 = 1.2566370614 * 1e-6  # 绝对磁化率 μ0
        # d=10**7                 #电导率 σ
        a1 = 1.38  # a=1.38
        # r=0.02                  #半径
        e = np.e
        global step
        step = 0

        def f(x, y):
            return np.tan(x) - ((y - 1) * x / ((y - 1) + x ** 2))  # 超越方程  先验方程

        def erfen(a, b, c):  # a、b为方程的根区间，c是相对磁化率
            global step
            fhalf = f((a + b) / 2, c)
            half = (a + b) / 2
            fa = f(a, c)
            fb = f(b, c)
            step = step + 1
            if (fa == 0):
                return a
            if (fb == 0):
                return a
            if (fhalf == 0):
                return fhalf
            if np.sqrt(abs(fa * fb)) < 1e-10:
                return a
            if fhalf * fb < 0:
                return erfen(half, b, c)
            else:
                return erfen(a, half, c)

        x = erfen(e1, e2, c)  # x即为超越方程的解  δ1
        # print(x)

        #################  t0 t1 ############
        t0 = (d * c * c0 * r ** 2) / (x ** 2)
        if c > 20:
            t1 = (d * c * c0 * r ** 2) / ((c + 2) * (c - 1))
        else:
            t1 = t0
        #################  K  #####################
        k = (6 * pi * r ** 3 * c) / (c + 2)
        #################  a  ###################
        a = a1 * t1
        #################  b  ######################
        b = 2 * (c + 2) * pow(a, 1 / 2) / pow((pi * c * c0 * d), 1 / 2) * r
        ##################  γ  ####################
        r1 = (1 + pow((a1 * t1 / 2 * t0), 1 / 2)) / (1 + pow((a1 * t1 / 2 * t0), 1 / 2) - b / 4)  # b
        R = r1 * t0

        # 椭球体改变了K
        return k, a, b, R

    def ellipsoid_k_plus(self, ta, tb, c):
        shape = ta / tb
        if shape < 1:  # h1，h2为退磁因子
            h1 = (shape ** 2 / (1 - shape ** 2)) * (
                    ((np.arctanh(pow(1 - shape ** 2, 1 / 2))) / pow(1 - shape ** 2, 1 / 2)) - 1)  # 轴向-对应b
            h2 = (1 / (2 * (1 - shape ** 2))) * (
                    1 - (shape ** 2 * np.arctanh(pow(1 - shape ** 2, 1 / 2)) / pow(1 - shape ** 2, 1 / 2)))  # 横向-对应a
        else:
            h1 = (shape ** 2 / (shape ** 2 - 1)) * (
                    1 - ((np.arctan(pow(shape ** 2 - 1, 1 / 2))) / pow(shape ** 2 - 1, 1 / 2)))
            h2 = (1 / (2 * (shape ** 2 - 1))) * (
                    (shape ** 2 * np.arctan(pow(shape ** 2 - 1, 1 / 2)) / pow(shape ** 2 - 1, 1 / 2)) - 1)

        k1_plus = ((2 * ta ** 2 * tb * (c + 2)) / (9 * tb ** 3 * c)) * (
                (1 / (1 - h1)) + ((c - 1) / (1 + h1 * (c - 1))))
        k2_plus = ((2 * ta ** 2 * tb * (c + 2)) / (9 * ta ** 3 * c)) * (
                (1 / (1 - h2)) + ((c - 1) / (1 + h2 * (c - 1))))
        return k1_plus, k2_plus

    def ellipsoid_parameter(self, c, c0, d, ta, tb):
        k1, a1, b1, R1 = self.parameter_sphere(c, c0, d, tb)
        k2, a2, b2, R2 = self.parameter_sphere(c, c0, d, ta)
        k1_plus, k2_plus = self.ellipsoid_k_plus(ta, tb, c)
        k1_ellipsoid = k1_plus * k1
        k2_ellipsoid = k2_plus * k2

        return k1_ellipsoid, a1, b1, R1, k2_ellipsoid, a2, b2, R2

    def func(self, t, k, a, b, R):  # 返回球体的响应方程，用于拟合算法，需要此方程
        e = np.e
        return k * pow((1 + pow(t / a, 1 / 2)), -b) * pow(e, -t / R)

    def wgn_one_npower(self, x, snr):
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(x ** 2) / len(x)
        npower = float(xpower / snr)
        return npower

    def dpred(self):
        t_split = self.survey.source.collection.t_split
        attribute = self.survey.source.target.attribute
        snr = self.survey.source.collection.SNR
        a_r_step = self.survey.source.target.a_r_step
        b_r_step = self.survey.source.target.b_r_step
        ta_min = self.survey.source.target.ta_min
        ta_max = self.survey.source.target.ta_max
        tb_min = self.survey.source.target.tb_min
        tb_max = self.survey.source.target.tb_max
        material_list = self.survey.source.target.material
        lable = []
        feature = []
        material_cnt = 0
        t = np.array(10 ** (np.linspace(-8, 1, t_split * 10)))
        sample_num = 0
        plot_flag = 0

        while material_cnt < len(material_list):
            c = attribute[material_cnt, 0]
            c0 = attribute[material_cnt, 1]
            d = attribute[material_cnt, 2]
            ta = ta_min
            while ta <= ta_max:
                tb = tb_min
                while tb <= tb_max:
                    if ta != tb:
                        if snr == None:
                            k1_ellipsoid, a1, b1, R1, k2_ellipsoid, a2, b2, R2 = self.ellipsoid_parameter(c, c0, d, ta,
                                                                                                          tb)
                            M1_without_noise = np.array(
                                list(map(partial(self.func, k=k1_ellipsoid, a=a1, b=b1, R=R1), t)))
                            M2_without_noise = np.array(
                                list(map(partial(self.func, k=k2_ellipsoid, a=a2, b=b2, R=R2), t)))
                            M = np.hstack((M1_without_noise, M2_without_noise))
                            # 保存响应曲线的数据
                            # if material_cnt == 0:
                            #     if (ta == 0.04) & (tb == 0.08):
                            #         pd.DataFrame([M1_without_noise, M2_without_noise, t],
                            #                      index=['M1', 'M2', 't']).to_csv('./response_curve.csv')
                            #         pd.DataFrame([[k1_ellipsoid, a1, b1, R1, k2_ellipsoid, a2, b2, R2]],
                            #                      columns=['k1_ellipsoid', 'a1', 'b1', 'R1', 'k2_ellipsoid', 'a2', 'b2',
                            #                               'R2']).to_csv(
                            #             './results/tdemResults/response_parameter.csv')
                        if snr != None:
                            k1_ellipsoid, a1, b1, R1, k2_ellipsoid, a2, b2, R2 = self.ellipsoid_parameter(c, c0, d, ta,
                                                                                                          tb)
                            M1_without_noise = np.array(
                                list(map(partial(self.func, k=k1_ellipsoid, a=a1, b=b1, R=R1), t)))
                            M2_without_noise = np.array(
                                list(map(partial(self.func, k=k2_ellipsoid, a=a2, b=b2, R=R2), t)))
                            M_noise = np.hstack((M1_without_noise, M2_without_noise))
                            noise_power = self.wgn_one_npower(x=M_noise, snr=snr)
                            M1 = M1_without_noise + np.random.randn(len(M1_without_noise)) * np.sqrt(noise_power)
                            M2 = M2_without_noise + np.random.randn(len(M2_without_noise)) * np.sqrt(noise_power)
                            M = np.hstack((M1, M2))
                            # 输出第400个样本的分布情况
                            plot_flag += 1
                            if plot_flag == 400:
                                data_collect = np.vstack((M1, M2, M1_without_noise, M2_without_noise, t))
                                data_collect = pd.DataFrame(data_collect,
                                                            index=['M1', 'M2', 'M1_WITHOUT', 'M2_WITHOUT', 't'])
                                sample = {'data': data_collect, 'M1': M1, 'M2': M2,
                                          'M1_without_noise': M1_without_noise,
                                          'M2_without_noise': M2_without_noise, 't': t, 'SNR': snr,
                                          'material': material_list[material_cnt],
                                          'ta': ta, 'tb': tb, }
                                # data_collect.to_csv(dir_selected + 'SNR=' + str(snr) + 'dB.csv')
                        material_flag = material_cnt
                        if ta > tb:
                            shape_flag = 0  # 径向大于轴向，扁椭球体
                        else:
                            shape_flag = 1  # 径向小于轴向，长椭球体
                        if sample_num == 0:
                            feature = M
                            lable = [[material_flag, shape_flag]]
                        else:
                            feature = np.vstack((feature, M))
                            lable = np.vstack((lable, [[material_flag, shape_flag]]))
                        sample_num += 1
                        print("这是第%d个样本" % sample_num, material_flag, shape_flag, snr)
                    tb += b_r_step
                ta += a_r_step
            material_cnt += 1
        feature_lable = np.hstack((lable, feature))
        return feature_lable, sample

