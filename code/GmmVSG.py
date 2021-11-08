# -*-coding=utf-8-*-
import sys

import pandas as pd
import numpy as np
import os

from pandas import DataFrame
from sklearn.mixture import GaussianMixture as GM_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def calculate_dis_of_tp_and_vp_single(single_point, batch_point):
    """计算单个样本点(N维)到空间内其它点的欧氏距离
    param:
        single_point: 单个点的坐标，（N,）
        batch_point:  其余点的坐标，(samples，N)
    return:
        res：numpy.ndarray,(samples,)
    """
    b_s = (batch_point - single_point) ** 2
    su = np.sum(b_s, axis=1).astype(float)
    res = np.sqrt(su)
    return res


def calculate_dis_of_tp_and_vp_batch(created_data, origin_data):
    """计算虚拟样本点和原始数据集的欧氏距离
    param:
        created_data: 虚拟样本点，（samples_vs, N）
        origin_data:  原始数据集，(samples_t，N)
    return:
        res：numpy.ndarray,(samples_vs,samples_t)
        res_sort_idx：numpy.ndarray,(samples_vs,samples_t)
    """
    res_sort_idx = np.zeros(shape=(len(created_data), len(origin_data)), dtype="int")
    res = np.zeros(shape=(len(created_data), len(origin_data)))
    for i in range(len(created_data)):
        res_temp = calculate_dis_of_tp_and_vp_single(created_data[i, :], origin_data)
        res[i] = res_temp
        res_temp_sort_idx = np.argsort(res_temp)
        res_sort_idx[i] = res_temp_sort_idx
    return res, res_sort_idx


class GmmVSG:

    def __init__(self, covariance_type='full', optimizer='aic', result_retain=2, seed=0):
        self.covariance_type = covariance_type
        self.optimizer = optimizer
        self.result_retain = result_retain
        self.random_seed = seed

        self.__gmmModel = None
        self.__pca1 = None
        self.__pca2 = None
        self.__col_names = None
        self.__targetName = None
        self.__X = None
        self.__y = None

    def fit(self, file_path, feature_start_idx, feature_end_idx, n_components, target_name=None):
        """获得生成模型
        必选参数
        ----------
        file_path:              原始数据的存放路径
        feature_start_idx:      原始数据特征空间起始列数
        feature_end_idx:        原始数据特征空间结束列数
        n_components:           基于网格搜索优化模型时的搜索列表

        可选参数
        ----------
        target_name：            特征列名称
        """
        if not os.path.exists(file_path):
            print("File path does not exist, please re-enter！")
            sys.exit(-1)
        df_data = pd.read_csv(file_path, delimiter=',')
        plt.rcParams['font.sans-serif'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False
        # 画出原始数据的pca2维分布
        if target_name is None:
            X = df_data.values
            self.__col_names = list(df_data.columns.values)
            pca_X = X.copy()
            self.__pca1 = PCA(n_components=2)
            self.__pca1.fit(pca_X)
            X_2D = self.__pca1.transform(pca_X)
            plt.figure()
            plt.scatter(X_2D[:, 0], X_2D[:, 1], color='blue')
            plt.xlabel('PCA-Dimension1')
            plt.ylabel('PCA-Dimension2')
            plt.savefig('origin.jpg')
            plt.show()
        else:
            self.__targetName = target_name
            X = df_data.values[:, feature_start_idx - 1:feature_end_idx]
            y = df_data[target_name].values
            self.__X = X
            self.__y = y
            temp_col_name = df_data.columns.values[feature_start_idx - 1:feature_end_idx]
            col_name = [i for i in temp_col_name]
            col_name.append(target_name)
            self.__col_names = col_name
            pca_X = np.hstack((X, y.reshape(-1, 1)))
            self.__pca2 = PCA(n_components=2)
            self.__pca2.fit(pca_X)
            X_2D = self.__pca2.transform(pca_X)
            plt.figure()
            plt.scatter(X_2D[:, 0], X_2D[:, 1], color='blue')
            plt.xlabel('PCA-Dimension1')
            plt.ylabel('PCA-Dimension2')
            plt.savefig('origin.jpg')
            plt.show()

        # 2. 求GMM参数
        models = [GM_model(n, covariance_type=self.covariance_type, random_state=self.random_seed).fit(X)
                  for n in n_components]
        best_model_index = 0
        if self.optimizer.lower() == 'aic':
            score_aic = [m.aic(X) for m in models]
            best_model_index = np.argmin(score_aic)
        elif self.optimizer.lower() == 'bic':
            score_bic = [m.bic(X) for m in models]
            best_model_index = np.argmin(score_bic)
        else:
            print("There is no optimizer of this type, please confirm again!")
            exit(-1)

        # 3. 扩充数据
        gmm = models[best_model_index]
        gmm.fit(X)
        self.__gmmModel = gmm

    def samples(self, number):
        """生成虚拟样本
        必选参数
        ----------
        number : 要扩充生成多少条样本数目
        """
        # 1.读数据
        # 数据可视化
        if self.__gmmModel is None:
            print("fit() is needed before samples()!")
            exit(-1)
        VX = self.__gmmModel.sample(number)[0]
        # 4.保存数据
        fmt = "%.0{}f".format(self.result_retain)
        if self.__targetName is None:
            np.savetxt("result.csv", VX, delimiter=',', fmt=fmt)
            VX_2D = self.__pca1.transform(VX)
            plt.scatter(VX_2D[:, 0], VX_2D[:, 1], color='red')
            plt.xlabel('PCA-Dimension1')
            plt.ylabel('PCA-Dimension2')
            plt.savefig('vsg.jpg')
            plt.show()
            data_frame = DataFrame(VX[0:10], columns=self.__col_names)
            return data_frame
        else:
            X, y = self.__X, self.__y
            s_scale = StandardScaler()
            s_scale.fit(X)
            standard_X = s_scale.transform(X)
            standard_VX = s_scale.transform(VX)
            _, sort_idx = calculate_dis_of_tp_and_vp_batch(standard_VX, standard_X)
            created_y = np.array([y[i[0]] for i in sort_idx])
            created_data = np.hstack((VX, created_y.reshape(-1, 1)))
            np.savetxt("result.csv", created_data, delimiter=',', fmt=fmt)
            VX_2D = self.__pca2.transform(created_data)
            plt.scatter(VX_2D[:, 0], VX_2D[:, 1], color='red')
            plt.xlabel('PCA-Dimension1')
            plt.ylabel('PCA-Dimension2')
            plt.savefig('vsg.jpg')
            plt.show()
            data_frame = DataFrame(created_data[0:10], columns=self.__col_names)
            return data_frame
