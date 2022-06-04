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

class logita():
    def __init__(self, X, y, labeled, budget, X_test, y_test):
        self.X = X
        self.y = y
        self.Ninit, self.nAtt = X.shape
        self.labels = sorted(np.unique(self.y))
        self.nClass = len(self.labels)
        self.X_test = X_test
        self.y_test = y_test
        self.labeled = list(deepcopy(labeled))
        self.n_theta = [i for i in range(self.nClass - 1)]
        self.theta = None
        self.unlabeled = self.initialization()
        self.budget = deepcopy(budget)
        self.budgetLeft = deepcopy(budget)
        self.pdist = pdist(self.X, metric="euclidean")
        self.dist_matrix = squareform(self.pdist)
        self.target = np.array([tar for tar in np.arange(self.labels[0], self.labels[-1], 1)])
        self.nClass = len(self.labels)
        self.nTar = self.nClass - 1
        self.ocModel = OrderedDict()
        self.beta_mat = self.initialization_beta_mat()
        self.tar_idx = None
        self.Kp = self.nTar * self.nAtt


    def initialization(self):
        unlabeled = [i for i in range(self.Ninit)]
        for idx in self.labeled:
            unlabeled.remove(idx)
        return unlabeled

    def initialization_beta_mat(self):
        for tar in self.target:
            self.ocModel[tar] = LogisticRegression(solver='newton-cg', penalty='l2')
        ##----------------（3）初始化分类器参数矩阵
        beta_mat = np.zeros((self.nAtt,self.nTar))
        ## 构造K-1个训练样本集
        train_ids_dict = OrderedDict()
        for tar in self.target:
            train_ids_dict[tar] = []
        for idx in self.labeled:
            if self.y[idx] == self.labels[0]:
                train_ids_dict[self.y[idx]].append(idx)
            elif self.y[idx] == self.labels[-1]:
                train_ids_dict[self.y[idx]-1].append(idx)
            else:
                train_ids_dict[self.y[idx]].append(idx)
                train_ids_dict[self.y[idx] - 1].append(idx)
        for tar in self.target:
            self.ocModel[tar].fit(X=self.X[train_ids_dict[tar]],y=self.y[train_ids_dict[tar]])
            beta_mat[:,tar] = self.ocModel[tar].coef_[0]
        return beta_mat

        ## 每选择一个新的样本并标记以后，就要执行升级beta_mat的操作
    def update_beta_mat(self):
        model = LogisticRegression(solver='newton-cg', penalty='l2')
        if self.tar_idx == None:
            pass
        else:
            print("当前选择样本=",self.tar_idx)
            tar_lab = self.y[self.tar_idx]
            if tar_lab == self.labels[0]:
                train_ids = []
                for idx in self.labeled:
                    lab = self.y[idx]
                    if lab == self.labels[0] or lab == self.labels[1]:
                        train_ids.append(idx)
                model.fit(X=self.X[train_ids],y=self.y[train_ids])
                self.beta_mat[:,0] = model.coef_[0]
            elif tar_lab == self.labels[-1]:
                train_ids = []
                for idx in self.labeled:
                    lab = self.y[idx]
                    if lab == self.labels[-1] or lab == self.labels[-2]:
                        train_ids.append(idx)
                model.fit(X=self.X[train_ids],y=self.y[train_ids])
                self.beta_mat[:,-1] = model.coef_[0]
            else:
                for tar in [tar_lab - 1, tar_lab]:
                    train_ids = []
                    for idx in self.labeled:
                        lab = self.y[idx]
                        if lab == tar or lab == tar + 1:
                            train_ids.append(idx)
                    model.fit(X=self.X[train_ids],y=self.y[train_ids])
                    self.beta_mat[:,tar] = model.coef_[0]

    def get_WH(self):
        predictor = self.X[self.labeled] @ self.beta_mat
        cumsum_predictor = np.cumsum(predictor,axis=1)
        accu_tmp = np.exp(cumsum_predictor)
        tmp = np.exp(predictor)
        theta = tmp / (1 + tmp)
        phi = accu_tmp / (1 + np.sum(accu_tmp,axis=1)).reshape(-1,1)
        W = np.zeros((self.Kp, self.Kp))
        H = np.zeros((self.Kp, self.Kp))
        for i, idx in enumerate(self.labeled):
            xi = self.X[idx]
            gram = np.outer(xi,xi)
            w = np.zeros((self.nTar,self.nTar))
            # h = np.zeros((self.nTar, self.nTar))
            for j in range(self.nTar-1):
                w[j,j+1] = - phi[i,j] * (1-theta[i,j]) * theta[i,j+1]
            w = w.T + w
            for j in range(self.nTar):
                w[j,j] = phi[i,j] * (1-theta[i,j])
                # h[j,j] = deepcopy(w[j,j])
            W += np.kron(w,gram)
            # H += np.kron(h,gram)
        return W

    def A_optimal_ord(self,W):
        predictor = self.X[self.unlabeled] @ self.beta_mat
        cumsum_predictor = np.cumsum(predictor,axis=1)
        accu_tmp = np.exp(cumsum_predictor)
        tmp = np.exp(predictor)
        theta = tmp / (1 + tmp)
        phi = accu_tmp / (1 + np.sum(accu_tmp,axis=1)).reshape(-1,1)
        trace_value = OrderedDict()
        for i, idx in enumerate(self.unlabeled):
            xi = self.X[idx]
            gram = np.outer(xi, xi)
            w = np.zeros((self.nTar, self.nTar))
            for j in range(self.nTar - 1):
                w[j, j + 1] = - phi[i, j] * (1 - theta[i, j]) * theta[i, j + 1]
            w = w.T + w
            for j in range(self.nTar):
                w[j,j] = phi[i,j] * (1 - theta[i,j])

            trace_value[idx] = np.trace(pinv(W + np.kron(w,gram)))
        self.tar_idx = min(trace_value,key=trace_value.get)


    def select(self):
        while self.budgetLeft > 0:
            self.update_beta_mat()
            W = self.get_WH()
            self.A_optimal_ord(W=W)
            print("当前选择样本在已标记样本集：", self.tar_idx in self.labeled)
            self.labeled.append(self.tar_idx)
            self.unlabeled.remove(self.tar_idx)
            self.budgetLeft -= 1
            print("选了一个")





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

            model = logita(X=X_train, y=y_train, labeled=labeled, budget=Budget, X_test=X_test, y_test=y_test)
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
        save_path = Path(r"E:\FFFFF\SelectedResult\LogitA")
        save_path = str(save_path.joinpath(name + ".xls"))
        workbook.save(save_path)


