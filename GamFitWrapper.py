from sklearn.model_selection import KFold
from pygam import LogisticGAM
from multilabel import ProbabilisticClassifierChain
import numpy as np
from sklearn.base import clone
import warnings
warnings.filterwarnings("ignore")

class GamFitWrapper:

    def __init__(self, n_splines=25, spline_order=3, n_splits=30, rank='median') -> None:
        self.n_splines = n_splines
        self.n_splits = n_splits
        self.spline_order = spline_order
        self.error = None
        self.rank_option = rank
        self.error = [[] for each in range(self.n_splits)]
        self.rank = []
        self.lambs = []
        self.model = None
        self.chain = None
        if n_splits < 2: raise Exception ('n_splits should at least 2')
        if rank not in ['median', 'mean']: raise Exception ('Invalid ranking method')

    def fit(self, X, y):
        """
        X, y should be dataframe. Apply 2 methods ranking here.
        """
        i = 0
        kf = KFold(self.n_splits, shuffle=True)
        # StrtifiedKFold can not splitted multi-dimensions in y. Not useful in this part. We try KFold cv.
        # In this probabilistic classifier, it is dangerous to have small number of data size. For example, if one observation is not found in training samples, it will raise errors.
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values #remove .values
            y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values
            if len(y.shape) == 1:
                # if multiple targets, use first lambda values
                lamb = LogisticGAM(n_splines=self.n_splines, spline_order=self.spline_order).gridsearch(X_train, y_train).lam[0]
                logisticFit = LogisticGAM(n_splines=self.n_splines, spline_order=self.spline_order, lam=lamb[0]*X_train.shape[1]).fit(X_train, y_train)
            else:
                lamb = LogisticGAM(n_splines=self.n_splines, spline_order=self.spline_order).gridsearch(X_train, y_train[:, 0]).lam[0]
                logisticFit = ProbabilisticClassifierChain(LogisticGAM(n_splines=self.n_splines, spline_order=self.spline_order, lam=lamb[0]*X_train.shape[1])).fit(X_train, y_train)
            self.lambs.append(lamb)

            if len(y.shape) == 1:
                y_pred = logisticFit.predict_proba(X_test)[y_test]#[np.arange(len(y_test)), y_test]
            else:
                y_pred = logisticFit.predict_proba(X_test)
            test_error = self.log_loss(y_test, y_pred)
            self.error[i].append(test_error)
            self.rank.append([sorted(self.error[i]).index(l) for l in self.error[i]])
            i += 1
        if self.rank_option =='median':
                self.rank = np.median(np.array(self.rank), axis=0)
        else:
            self.rank = np.mean(np.array(self.rank), axis=0)
        lst = list(self.rank)
        indx = lst.index(min(lst))
        logisticG = LogisticGAM(n_splines=self.n_splines, spline_order=self.spline_order, lam=self.lambs[indx]*X_train.shape[1]).fit(X_train, y_train)
        if len(y.shape) == 1:
            # self.chain = rf
            self.model = clone(logisticG)
            self.model.fit(X.values, y.values) # change to matrix
        else:
            self.chain = ProbabilisticClassifierChain(clone(logisticG))
            self.chain.fit(X, y)
        return self


    def predict_proba(self, X):
        """This is only for predicting probability of p(y=1).
        """
        if self.chain:
            return self.chain.predict_proba(X)
        return self.model.predict_proba(X.values)

    def predict(self, X):
        """This is for converting probability to binary (0, 1)
        if p>0.5, label=1, otherwise 0
        """
        if self.chain:
            result = self.predict_proba(X)
            result[result>=0.5] = 1
            result[result<0.5] = 0
        else:
            result = self.model.predict(X.values)
        return result

    def log_loss(self, y_test, y_pred):
        y_test = y_test.astype(np.float16)
        y_pred = y_pred.astype(np.float16)
        if len(y_test.shape) == 1:
            N = y_test.shape[0]
            loss = 0
            for i in range(N):
                loss -= ((y_test[i]*np.log2(y_pred[i]))+((1.0-y_test[i])*np.log2(1.0-y_pred[i])))
                loss = loss/N
        else:
            N,M = y_test.shape
            a=[]
            for m in range(M):
                loss=0
                for i in range(N):
                    loss -= ((y_test[i,m]*np.log2(y_pred[i,m]))+((1.0-y_test[i,m])*np.log2(1.0-y_pred[i,m])))
                loss = loss/N
                a.append(round(loss,8))
            loss = np.mean(a)
        return loss