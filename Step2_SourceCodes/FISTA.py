import xlwt
import xlrd
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from collections import OrderedDict
from sklearn.svm import SVC
from scipy.special import expit
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from KBS_NEW.PointwiseQuery.ALOR import ALOR
from sklearn.metrics import accuracy_score, mean_squared_error
from time import time
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y
from scipy.linalg import pinv, pinv2, pinvh
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

class FistaLocp():
    def __init__(self,X_pool,y_pool,labeled,budget,X_test,y_test):
        self.X_pool = X_pool
        self.y_pool = y_pool.astype(np.int32)
        self.X_test = X_test
        self.y_test = y_test.astype(np.int32)
        self.classNum = len(set(self.y_pool))
        self.labeled = list(deepcopy(labeled))
        self.budgetLeft = deepcopy(budget)
        self.budget = deepcopy(budget)
        self.FSITA_selected = self.fistlocp()
        self.theta = None
        self.w = None
        self.unlabeled = self.initialization()


    def initialization(self):
        unlabeled = [i for i in range(len(self.y_pool))]
        for j in self.labeled:
            unlabeled.remove(j)
        return unlabeled


    def t_updata(self, t_old):
        return .5 + 0.5 * np.sqrt(1 + 4 * t_old)
    def C_cal(self, W, X, gamma):
        return W - gamma * (X.T * X * W - X.T * X)
    def P_zeta_1(self, a, b, zeta):
        a_abs = np.abs(sorted(a, reverse=True))
        t = 0
        a_abs_temp = np.append(a_abs, 0)
        for j in range(0, 3):  # range(len(a)):
            t = t + a_abs[j]
            delta = (t - b) / (pow(zeta, -1) + j + 1)
            if a_abs_temp[j + 1] <= delta <= a_abs_temp[j]:
                a_abs_delta = np.abs(a) - delta
                # a_abs_delta = a_abs - delta
                Zero = np.zeros(len(a_abs))
                a_abs_delta = np.maximum(Zero, a_abs_delta)
                X = np.sign(a) * a_abs_delta
                y = np.linalg.norm(X, 1)
                return X, y
        X = a
        y = b
        return X, y

    def KMeans_results(self, X, Cluster_Num):
        kmeans = KMeans(n_clusters=Cluster_Num).fit(X)
        clusterIndex = kmeans.labels_
        clusterDict = OrderedDict()
        for k in range(Cluster_Num):
            clusterDict[k] = []
        for i in range(len(X)):
            clusterDict[clusterIndex[i]].append(i)
        return clusterDict

    def Problem(self, X, W_old, G, gamma):
        t_old = 1
        W_current = deepcopy(W_old)
        W_new = np.zeros((len(W_old), len(W_old)))
        iter_num = 10
        while iter_num > 0:
            # print("迭代剩余次数=",iter_num)
            C = self.C_cal(W=W_current, X=X, gamma=gamma)  # 调用函数计算大C，d(*) 为求导; C = W - r* d(F(W))
            for g in G.values():  # 对每个类簇执行以下运算
                T = np.zeros(len(g))
                for i, gi in enumerate(g):
                    T[i] = np.linalg.norm(C[gi], 2)
                S, y = self.P_zeta_1(a=T, b=0, zeta=1)
                for i, gi in enumerate(g):
                    W_new[gi, :] = (S[i] / T[i]) * C[gi, :]

            t_new = self.t_updata(t_old)
            # print("t_new = ",t_new)
            W_current = W_new + (W_new - W_old) * (t_old - 1) / t_new
            W_old = deepcopy(W_new)
            t_old = deepcopy(t_new)
            iter_num -= 1
        return W_new

    def fistlocp(self):
        ClustG = self.KMeans_results(self.X_pool, Cluster_Num=self.classNum)
        train_XX = np.mat(self.X_pool).T
        N = train_XX.shape[1]
        W_old = np.random.random((N, N)) - 0.5
        W_new = self.Problem(X=train_XX, W_old=W_old, G=ClustG, gamma=1e-06)
        row_sum_abs = np.linalg.norm(W_new, ord=1, axis=1)
        Ord_rank = np.flipud(np.argsort(row_sum_abs))
        selected = Ord_rank[0:(self.budgetLeft + 20)]
        return list(selected)



    def select(self):
        while self.budgetLeft > 0:
            idx = self.FSITA_selected[0]
            if idx in self.labeled:
                self.FSITA_selected.remove(idx)
                continue
            else:
                self.labeled.append(idx)
                self.unlabeled.remove(idx)
                self.FSITA_selected.remove(idx)
                self.budgetLeft -= 1



if __name__ == '__main__':
    names_list = ["Balance-scale", "HDI","Car","ARWU2020","Bank-5bin","Computer-5bin","Automobile","Obesity","Bank-10bin","Computer-10bin","PowerPlant"]

    for name in names_list:
        print("########################{}".format(name))
        p = Path("D:\OCdata")
        data_path = Path(r"D:\OCdata")
        partition_path = Path(r"E:\FFFFF\DataPartitions")
        # kmeans_path = Path(r"E:\CCCCC_Result\KmeansResult")
        """--------------read the whole data--------------------"""
        read_data_path = data_path.joinpath(name + ".csv")
        data = np.array(pd.read_csv(read_data_path, header=None))
        X = np.asarray(data[:, :-1], np.float64)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = data[:, -1]
        y -= y.min()
        nClass = len(np.unique(y))
        Budget = 10 * nClass

        """--------read the partitions--------"""
        read_partition_path = str(partition_path.joinpath(name + ".xls"))
        book_partition = xlrd.open_workbook(read_partition_path)

        """-----read the kmeans results according to the partition-----"""
        # read_kmeans_path = str(kmeans_path.joinpath(name + ".xls"))
        # book_kmeans = xlrd.open_workbook(read_kmeans_path)
        workbook = xlwt.Workbook()
        count = 0
        for SN in book_partition.sheet_names():
            S_Time = time()
            train_idx = []
            test_idx = []
            labeled = []
            table_partition = book_partition.sheet_by_name(SN)
            for idx in table_partition.col_values(0):
                if isinstance(idx,float):
                    train_idx.append(int(idx))
            for idx in table_partition.col_values(1):
                if isinstance(idx,float):
                    test_idx.append(int(idx))
            for idx in table_partition.col_values(2):
                if isinstance(idx,float):
                    labeled.append(int(idx))

            X_train = X[train_idx]
            y_train = y[train_idx].astype(np.int32)
            X_test = X[test_idx]
            y_test = y[test_idx]

            model = FistaLocp(X_pool=X_train, y_pool=y_train, labeled=labeled, budget=Budget, X_test=X_test, y_test=y_test)
            model.select()
            # SheetNames = "{}".format(count)
            sheet = workbook.add_sheet(SN)
            for i, idx in enumerate(train_idx):
                sheet.write(i, 0,  int(idx))
            for i, idx in enumerate(test_idx):
                sheet.write(i, 1, int(idx))
            for i, idx in enumerate(labeled):
                sheet.write(i, 2, int(idx))
            for i, idx in enumerate(model.labeled):
                sheet.write(i, 3, int(idx))

            print("SN:",SN," Time:",time()-S_Time)
        save_path = Path(r"E:\FFFFF\SelectedResult\FISTA")
        save_path = str(save_path.joinpath(name + ".xls"))
        workbook.save(save_path)


