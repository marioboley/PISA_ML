from realkd.patch import RuleFit
from sklearn.model_selection import KFold
from multilabel import ProbabilisticClassifierChain
import numpy as np
from sklearn.base import clone
import re
from sklearn.metrics import log_loss

class RuleFitWrapper:

    def __init__(self, Cs = [1, 2, 4, 8, 16, 32], n_splits=10, rank='median', mode='pcc'):
        """
        Input: Cs: C candidates list, orginal Cs is [0.1, 0.5, 1, 2, 4, 8, 16, 32]. To save time, we get rid of 0.1 and 0.5
               n_splits: default is 10 Folder cross validation, if n_splits = n, leave one out cross validation, 
                        Choose large number of splits are safety for probabiltic classifier in multi-label classification.
               rank: selecting C criteria, 'median' or 'mean'
        """
        self.cs = Cs
        self.n_splits = n_splits
        self.rank = []
        self.error = [[] for _ in range(n_splits)]
        self.rank_option = rank
        self.chain = None
        self.rf = None
        self.num_rules = [[] for _ in range(n_splits)]
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
                # if multiple targets, use chain rules
                rulefits = [RuleFit(rfmode='classify', model_type='lr', Cs=[C]) for C in self.cs]
            else:
                rulefits = [ProbabilisticClassifierChain(RuleFit(rfmode='classify', model_type='lr', Cs=[C])) for C in self.cs]
            for each in rulefits:
                # SSE
                # each.fit(X_train, y_train)
                # try:
                #     test_error = sum(sum((y_test - each.predict(X_test))**2))/len(X_test)
                # except:
                #     test_error = sum((y_test - each.predict(X_test))**2)/len(X_test)
                each.fit(X_train, y_train)
                if len(y.shape) == 1:
                    y_pred = each.predict_proba(X_test)[np.arange(len(y_test)), y_test]
                else:
                    y_pred = each.predict_proba(X_test)
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
        rf = RuleFit(rfmode='classify', model_type='r', Cs=[self.cs[indx]])
        if len(y.shape) == 1:
            # self.chain = rf
            self.rf = clone(rf)
            self.rf.fit(X.values, y.values) # change to matrix
        else:
            self.chain = ProbabilisticClassifierChain(clone(rf))
            self.chain.fit(X,y)
        return self

    def predict_proba(self, X):
        """This is only for predicting probability of p(y=1).
        """
        if self.chain:
            return self.chain.predict_proba(X)
        return self.rf.predict_proba(X.values)

    def predict(self, X):
        """This is for converting probability to binary (0, 1)
        if p>0.5, label=1, otherwise 0
        """
        if self.chain:
            result = self.predict_proba(X)
            result[result>=0.5] = 1
            result[result<0.5] = 0
        else:
            result = self.rf.predict(X.values)
        return result

    def format_rules(self, feature_names, data):
        """This is for format the rule features
        Input: feature_names: feature names
               data: dataframe
        """
        dic_rules = dict(zip(['feature_' + str(i) + ' ' for i in range(len(feature_names))], feature_names))
        for key in dic_rules:
            if re.match(r'(.*?)' + key, data):
                data = re.sub(key, dic_rules[key], data)
        return data

    def get_rules(self, X, y, predictors=None):
        """Get positive rules
        Input: X: features
               y: targets
        """
        try:
            y_col = y.columns.tolist()
        except:
            y_col = [y.name]
        sum_columns = predictors + y_col
        res = {}
        if self.chain:
            rf = self.chain.fitted_
        else:
            rf = [self.rf]
        for i, est in enumerate(rf):
            rules = est.get_rules()
            rules = rules[rules['coef'] != 0]
            rules = rules.iloc[np.argsort(rules['importance'])][::-1]
            rules['rule'] = rules.apply(lambda x: self.format_rules(sum_columns, x.rule), axis=1)
            rules = rules.reset_index(drop=True)
            res[y_col[i]] = rules
        return res

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