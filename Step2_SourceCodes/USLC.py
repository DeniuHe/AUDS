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
from sklearn.linear_model import LogisticRegression



class RED_logist():
    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.int32)
        self.nSample, self.nDim = X.shape
        self.labels = list(np.sort(np.unique(y)))
        self.nClass = len(self.labels)
        self.nTheta = self.nClass - 1
        self.extend_part = np.eye(self.nClass-1)
        self.label_dict = self.get_label_dict()
        self.eX, self.ey = self.train_set_construct(X=self.X, y=self.y)
        self.model = LogisticRegression()
        self.model.fit(X=self.eX, y=self.ey)
        return self

    def get_label_dict(self):
        label_dict = OrderedDict()
        for i, lab in enumerate(self.labels):
            tmp_label = np.ones(self.nTheta)
            for k, pad in enumerate(self.labels[:-1]):
                if lab <= pad:
                    tmp_label[k] = 1
                else:
                    tmp_label[k] = -1
            label_dict[lab] = tmp_label
        return label_dict

    def train_set_construct(self, X, y):
        eX = np.zeros((self.nSample * self.nTheta, self.nDim + self.nTheta))
        ey = np.zeros(self.nSample * self.nTheta)

        for i in range(self.nSample):
            eXi = np.hstack((np.tile(X[i], (self.nTheta, 1)), self.extend_part))
            eX[self.nTheta * i: self.nTheta * i + self.nTheta] = eXi
            ey[self.nTheta * i: self.nTheta * i + self.nTheta] = self.label_dict[y[i]]
        return eX, ey

    def test_set_construct(self, X_test):
        nTest = X_test.shape[0]
        eX = np.zeros((nTest * self.nTheta, self.nDim + self.nTheta))
        for i in range(nTest):
            eXi = np.hstack((np.tile(X_test[i],(self.nTheta,1)), self.extend_part))
            eX[self.nTheta * i: self.nTheta * i + self.nTheta] = eXi
        return eX

    def predict(self, X):
        nTest = X.shape[0]
        eX_test = self.test_set_construct(X_test=X)
        y_extend = self.model.predict(X=eX_test)

        y_tmp = y_extend.reshape(nTest,self.nTheta)
        y_pred = np.sum(y_tmp < 0, axis=1).astype(np.int32)
        return y_pred

    def predict_proba(self, X):
        nTest = X.shape[0]
        eX_test = self.test_set_construct(X_test=X)
        dist_tmp = self.model.decision_function(X=eX_test)
        dist_matrix = dist_tmp.reshape(nTest, self.nTheta)
        accumulative_proba = expit(dist_matrix)
        prob = np.pad(
            accumulative_proba,
            pad_width=((0, 0), (1, 1)),
            mode='constant',
            constant_values=(0, 1))
        prob = np.diff(prob)
        return prob

    def distant_to_theta(self, X):
        nTest = X.shape[0]
        eX_test = self.test_set_construct(X_test=X)
        dist_tmp = self.model.decision_function(X=eX_test)
        dist_matrix = dist_tmp.reshape(nTest, self.nTheta)
        return dist_matrix


class USLC():
    def __init__(self, X, y, labeled, budget, X_test, y_test):
        self.X = X
        self.y = y
        self.nSample, self.nDim = X.shape
        self.labels = sorted(np.unique(self.y))
        self.nClass = len(self.labels)
        self.X_test = X_test
        self.y_test = y_test
        self.labeled = list(deepcopy(labeled))
        self.model = RED_logist()
        self.n_theta = [i for i in range(self.nClass - 1)]
        self.theta = None
        self.unlabeled = self.initialization()
        self.budget = deepcopy(budget)
        self.budgetLeft = deepcopy(budget)

    def initialization(self):
        unlabeled = list(range(self.nSample))
        for idx in self.labeled:
            unlabeled.remove(idx)
        self.model.fit(self.X[self.labeled], self.y[self.labeled])
        return unlabeled

    def evaluation(self):
        self.model.fit(self.X[self.labeled], self.y[self.labeled])

    def select(self):
        while self.budgetLeft > 0:
            prob_matrix = self.model.predict_proba(self.X[self.unlabeled])
            max_prob_list = np.max(prob_matrix, axis=1)
            ordidx = np.argsort(max_prob_list)
            tar_idx = self.unlabeled[ordidx[0]]
            self.unlabeled.remove(tar_idx)
            self.labeled.append(tar_idx)
            self.budgetLeft -= 1
            self.evaluation()



if __name__ == '__main__':
    names_list = ["Balance-scale", "HDI","Car","ARWU2020","Bank-5bin","Computer-5bin","Automobile","Obesity","Bank-10bin","Computer-10bin","PowerPlant"]


    for name in names_list:
        print("########################{}".format(name))

        data_path = Path(r"D:\OCdata")
        partition_path = Path(r"E:\FFFFF\DataPartitions")

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

            model = USLC(X=X_train, y=y_train, labeled=labeled, budget=Budget, X_test=X_test, y_test=y_test)
            model.select()

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
        save_path = Path(r"E:\FFFFF\SelectedResult\USLC")
        save_path = str(save_path.joinpath(name + ".xls"))
        workbook.save(save_path)
