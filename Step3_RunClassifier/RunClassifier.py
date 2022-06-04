import pandas as pd
import numpy as np
import xlrd
import xlwt
from time import time
from pathlib import Path
from collections import OrderedDict
from scipy.special import expit
from pathlib import Path
from numpy.linalg import inv
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, mutual_info_score
from sklearn.linear_model import LogisticRegression
from scipy.linalg import pinv


class RED_logist(ClassifierMixin, BaseEstimator):
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



# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class results():
    def __init__(self):
        self.MZEList = []
        self.MAEList = []
        self.F1List = []
        self.MIList = []
        self.ALC_MZE = []
        self.ALC_MAE = []
        self.ALC_F1 = []
        self.ALC_MI = []

class stores():
    def __init__(self):
        self.MZEList_mean = []
        self.MZEList_std = []
        # -----------------
        self.MAEList_mean = []
        self.MAEList_std = []
        # -----------------
        self.F1List_mean = []
        self.F1List_std = []
        # -----------------
        self.MIList_mean = []
        self.MIList_std = []
        # -----------------
        self.ALC_MZE_mean = []
        self.ALC_MZE_std = []
        # -----------------
        self.ALC_MAE_mean = []
        self.ALC_MAE_std = []
        # -----------------
        self.ALC_F1_mean = []
        self.ALC_F1_std = []
        # -----------------
        self.ALC_MI_mean = []
        self.ALC_MI_std = []
        # -----------------
        self.ALC_MZE_list = []
        self.ALC_MAE_list = []
        self.ALC_F1_list = []
        self.ALC_MI_list = []

# --------------------------------------

def get_train_test_init_selected_ids(name,method,result_path,data_path, save_path):
    read_data_path = data_path.joinpath(name + ".csv")
    data = np.array(pd.read_csv(read_data_path, header=None))
    X = np.asarray(data[:, :-1], np.float64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = data[:, -1]
    y -= y.min()
    read_path = str(result_path.joinpath(name + ".xls"))
    book = xlrd.open_workbook(read_path)
    RESULT = results()
    STORE = stores()

    for SN in book.sheet_names():
        S_time = time()   # -------record the time consumption---------
        table = book.sheet_by_name(SN)
        train_idx = []
        test_idx = []
        init_idx = []
        selected_idx = []
        for idx in table.col_values(0):
            if isinstance(idx,float):
                train_idx.append(int(idx))
        for idx in table.col_values(1):
            if isinstance(idx,float):
                test_idx.append(int(idx))
            else:
                break
        for idx in table.col_values(2):
            if isinstance(idx,float):
                init_idx.append(int(idx))
            else:
                break
        for idx in table.col_values(3):
            if isinstance(idx,float):
                selected_idx.append(int(idx))
            else:
                break
        # ---------------------------------
        X_pool = X[train_idx]
        y_pool = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        # ---------------------------------
        MZE_list = []
        MAE_list = []
        F1_list = []
        MI_list = []
        ALC_MZE = 0.0
        ALC_MAE = 0.0
        ALC_F1 = 0.0
        ALC_MI = 0.0
        # ----------------------------------------------------------- #
        # ---------------------Run Classifier------------------------ #
        # ----------------------------------------------------------- #
        init_len = len(init_idx)
        whole_len = len(selected_idx)
        for i in range(init_len,whole_len):
            current_ids = selected_idx[:i]
            # --------------------------------------
            model = RED_logist()
            model.fit(X=X_pool[current_ids], y=y_pool[current_ids])
            y_hat = model.predict(X=X_test)
            # --------------------------------------
            # --------------Metrics-----------------
            MZE = 1 - accuracy_score(y_hat, y_test)
            MAE = mean_absolute_error(y_hat, y_test)
            F1 = f1_score(y_pred=y_hat, y_true=y_test,average="macro")
            MI = mutual_info_score(labels_true=y_test, labels_pred=y_hat)

            # -------------------------------------
            MZE_list.append(MZE)
            MAE_list.append(MAE)
            F1_list.append(F1)
            MI_list.append(MI)
            ALC_MZE += MZE
            ALC_MAE += MAE
            ALC_F1 += F1
            ALC_MI += MI
            # -------------------------------------
        RESULT.MZEList.append(MZE_list)
        RESULT.MAEList.append(MAE_list)
        RESULT.F1List.append(F1_list)
        RESULT.MIList.append(MI_list)
        #-----------------------------
        RESULT.ALC_MZE.append(ALC_MZE)
        RESULT.ALC_MAE.append(ALC_MAE)
        RESULT.ALC_F1.append(ALC_F1)
        RESULT.ALC_MI.append(ALC_MI)


        print("==========time:{}".format(time()-S_time))

    STORE.MZEList_mean = np.mean(RESULT.MZEList, axis=0)
    STORE.MZEList_std = np.std(RESULT.MZEList, axis=0)
    STORE.MAEList_mean = np.mean(RESULT.MAEList, axis=0)
    STORE.MAEList_std = np.std(RESULT.MAEList, axis=0)
    STORE.F1List_mean = np.mean(RESULT.F1List, axis=0)
    STORE.F1List_std = np.std(RESULT.F1List, axis=0)
    STORE.MIList_mean = np.mean(RESULT.MIList, axis=0)
    STORE.MIList_std = np.std(RESULT.MIList, axis=0)
    # ------------------------------
    STORE.ALC_MZE_mean = np.mean(RESULT.ALC_MZE)
    STORE.ALC_MZE_std = np.std(RESULT.ALC_MZE)
    STORE.ALC_MAE_mean = np.mean(RESULT.ALC_MAE)
    STORE.ALC_MAE_std = np.std(RESULT.ALC_MAE)
    STORE.ALC_F1_mean = np.mean(RESULT.ALC_F1)
    STORE.ALC_F1_std = np.std(RESULT.ALC_F1)
    STORE.ALC_MI_mean = np.mean(RESULT.ALC_MI)
    STORE.ALC_MI_std = np.std(RESULT.ALC_MI)
    # ------------------------------------
    STORE.ALC_MZE_list = RESULT.ALC_MZE
    STORE.ALC_MAE_list = RESULT.ALC_MAE
    STORE.ALC_F1_list = RESULT.ALC_F1
    STORE.ALC_MI_list = RESULT.ALC_MI

    # -------------------------------------------------------
    sheet_names = ["MZE_mean", "MZE_std", "MAE_mean", "MAE_std", "F1_mean", "F1_std", "MI_mean", "MI_std", "NMI_mean", "NMI_std",
                   "ALC_MZE_list","ALC_MAE_list","ALC_F1_list","ALC_MI_list","ALC_MZE", "ALC_MAE", "ALC_F1", "ALC_MI", "ALC_NMI"]
    workbook = xlwt.Workbook()
    for sn in sheet_names:
        print("sn::",sn)
        sheet = workbook.add_sheet(sn)
        n_col = len(STORE.MZEList_mean)
        if sn == "MZE_mean":
            sheet.write(0, 0, method)
            for j in range(1,n_col + 1):
                sheet.write(0,j,STORE.MZEList_mean[j - 1])
        elif sn == "MZE_std":
            sheet.write(0, 0, method)
            for j in range(1,n_col + 1):
                sheet.write(0,j,STORE.MZEList_std[j - 1])
        elif sn == "MAE_mean":
            sheet.write(0, 0, method)
            for j in range(1,n_col + 1):
                sheet.write(0,j,STORE.MAEList_mean[j - 1])
        elif sn == "MAE_std":
            sheet.write(0, 0, method)
            for j in range(1,n_col + 1):
                sheet.write(0,j,STORE.MAEList_std[j - 1])
        elif sn == "F1_mean":
            sheet.write(0, 0, method)
            for j in range(1,n_col + 1):
                sheet.write(0,j,STORE.F1List_mean[j - 1])
        elif sn == "F1_std":
            sheet.write(0, 0, method)
            for j in range(1,n_col + 1):
                sheet.write(0,j,STORE.F1List_std[j - 1])

        elif sn == "MI_mean":
            sheet.write(0, 0, method)
            for j in range(1,n_col + 1):
                sheet.write(0,j,STORE.MIList_mean[j - 1])
        elif sn == "MI_std":
            sheet.write(0, 0, method)
            for j in range(1,n_col + 1):
                sheet.write(0,j,STORE.MIList_std[j - 1])
        # ---------------------------------------------------
        elif sn == "ALC_MZE_list":
            sheet.write(0, 0, method)
            for j in range(1,len(STORE.ALC_MZE_list) + 1):
                sheet.write(0,j,STORE.ALC_MZE_list[j - 1])
        elif sn == "ALC_MAE_list":
            sheet.write(0, 0, method)
            for j in range(1,len(STORE.ALC_MAE_list) + 1):
                sheet.write(0,j,STORE.ALC_MAE_list[j - 1])
        elif sn == "ALC_F1_list":
            sheet.write(0, 0, method)
            for j in range(1,len(STORE.ALC_F1_list) + 1):
                sheet.write(0,j,STORE.ALC_F1_list[j - 1])
        elif sn == "ALC_MI_list":
            sheet.write(0, 0, method)
            for j in range(1,len(STORE.ALC_MI_list) + 1):
                sheet.write(0,j,STORE.ALC_MI_list[j - 1])
        # -----------------
        elif sn == "ALC_MZE":
            sheet.write(0, 0, method)
            sheet.write(0, 1, STORE.ALC_MZE_mean)
            sheet.write(0, 2, STORE.ALC_MZE_std)
        elif sn == "ALC_MAE":
            sheet.write(0, 0, method)
            sheet.write(0, 1, STORE.ALC_MAE_mean)
            sheet.write(0, 2, STORE.ALC_MAE_std)
        elif sn == "ALC_F1":
            sheet.write(0, 0, method)
            sheet.write(0, 1, STORE.ALC_F1_mean)
            sheet.write(0, 2, STORE.ALC_F1_std)

        elif sn == "ALC_MI":
            sheet.write(0, 0, method)
            sheet.write(0, 1, STORE.ALC_MI_mean)
            sheet.write(0, 2, STORE.ALC_MI_std)

    save_path = str(save_path.joinpath(method))
    save_path = Path(save_path)
    save_path = str(save_path.joinpath(name + ".xls"))
    workbook.save(save_path)


if __name__ == '__main__':

    names_list = ["Balance-scale", "HDI","Car","ARWU2020","Bank-5bin","Computer-5bin","Automobile","Obesity","Bank-10bin","Computer-10bin","PowerPlant"]

    method_list = ["Random","USME","USLC","USMS","FISTA","ALCE","McPAL","MCSVMA","ALOR","LogitA","ADUS"]


    data_path = Path(r"D:\OCdata")
    for method in methods_list:
        print("@@@@@@@@@@@@@@@@@@@@@@{}".format(method))
        result_path = Path(r"E:\FFFFF\SelectedResult")
        save_path = Path(r"E:\FFFFF\ALResult")
        result_path = str(result_path.joinpath(method))
        result_path = Path(result_path)
        for name in names_list:
            print("@@@@@@@@@@@@@@@@@@@@@@{}".format(name))
            get_train_test_init_selected_ids(name,method,result_path,data_path,save_path)



