import xlwt
import xlrd
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
from time import time
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances

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
        self.lamb = 0.01
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

    # --------把需要计算的准则放在分类模型中-------------
    def MMC(self, X):
        nTest = X.shape[0]
        eX_test = self.test_set_construct(X_test=X)
        proba_matrix = self.model.predict_proba(X=eX_test)
        proba_list = proba_matrix[:,0] * proba_matrix[:,1]
        proba_list_reshape = proba_list.reshape(nTest, self.nTheta)

        eX_norm = np.linalg.norm(eX_test, ord=2, axis=1)
        eX_norm_reshape = eX_norm.reshape(nTest, self.nTheta)

        mmc_tmp = proba_list_reshape * eX_norm_reshape
        return np.sum(mmc_tmp, axis=1)

    def MSEE(self, X, XU):
        # nTest = X.shape[0]
        # ---------Fisher Matrix--------------------------
        eX_test = self.test_set_construct(X_test=X)
        proba_matrix = self.model.predict_proba(X=eX_test)
        proba_list = proba_matrix[:,0] * proba_matrix[:,1]
        # print("proba_list::",proba_list.shape)
        self.XWX = eX_test.T @ np.diag(proba_list) @ eX_test
        # print("===",self.XWX.shape)
        self.FM = pinv(self.XWX + self.lamb * np.eye(self.XWX.shape[0]))

        # --------Bias2------------------------------------
        eXU = self.test_set_construct(X_test=XU)
        proba_matrix = self.model.predict_proba(X=eXU)
        proba_list = proba_matrix[:,0] * proba_matrix[:,1]
        Bias2 = (self.lamb * proba_list * (eXU @ self.FM @ self.model.coef_[0])) ** 2

        # -------Var--------------------------------------
        Var = np.zeros(eXU.shape[0])
        MidPart = self.FM @ self.XWX @ self.FM
        proba_list_2 = proba_list ** 2
        for i in range(eXU.shape[0]):
            Var[i] = proba_list_2[i] * eXU[i] @ MidPart @ eXU[i].T
            # print("Var::",Var[i])
        # -----------------MSEE--------------------------
        MSEE = np.sum(Bias2 + Var)

        return MSEE





class MSMMCEER():
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
        self.pdist = pdist(self.X, metric="euclidean")
        self.dist_matrix = squareform(self.pdist)

    def initialization(self):
        unlabeled = [i for i in range(self.nSample)]
        for idx in self.labeled:
            unlabeled.remove(idx)
        self.model.fit(self.X[self.labeled], self.y[self.labeled])
        return unlabeled

    def evaluation(self):
        self.model.fit(self.X[self.labeled], self.y[self.labeled])

    def select(self):
        while self.budgetLeft > 0:
            for k in self.n_theta:
                if self.budgetLeft <= 0:
                    break
                # print("k::::",k)
                # ----------unlabeled instance close to theta_k-----------
                dist_matrix = abs(self.model.distant_to_theta(X=self.X[self.unlabeled]))
                ord_k_min_dist = np.argmin(dist_matrix, axis=1)
                # print("ord_k_min_dist::",len(ord_k_min_dist))
                ord_idx_unlabeled = np.where(ord_k_min_dist==k)[0]
                # print("ord_idx_unlabeled::",len(ord_idx_unlabeled))
                candi_unlabeled = np.asarray(self.unlabeled)[ord_idx_unlabeled]
                if candi_unlabeled.any():
                    candi_dist = dist_matrix[:,k][ord_idx_unlabeled]

                else:
                    # print("*******************************************")
                    continue
                # print("candi_unlabeled::",len(candi_unlabeled))
                # -----------Diversity------------------------------------
                dist_M = pairwise_distances(X=self.X[candi_unlabeled], Y=self.X[self.labeled], metric='euclidean')
                Div = np.min(dist_M, axis=1)

                # ----------Candidate--------------------------------------
                Candidate = []
                B = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
                for b in B:
                    Metric = OrderedDict()
                    for i, idx in enumerate(candi_unlabeled):
                        Metric[idx] = Div[i] ** b / (candi_dist[i]+1) ** (1-b)
                    tar_idx = max(Metric, key=Metric.get)
                    Candidate.append(tar_idx)
                # --------adaptive combination----------------------------
                MSEE = OrderedDict()
                for idx in Candidate:
                    tmp_labeled = deepcopy(self.labeled)
                    tmp_labeled.append(idx)
                    MSEE[idx] = self.model.MSEE(X=self.X[tmp_labeled],XU=self.X[self.unlabeled])

                tar_idx = min(MSEE, key=MSEE.get)
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

            model = MSMMCEER(X=X_train, y=y_train, labeled=labeled, budget=Budget, X_test=X_test, y_test=y_test)
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
        save_path = Path(r"E:\FFFFF\SelectedResult\proposed")
        save_path = str(save_path.joinpath(name + ".xls"))
        workbook.save(save_path)




