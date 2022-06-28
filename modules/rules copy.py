from ast import Return
from realkd.patch import RuleFit
from sklearn.model_selection import KFold
import numpy as np
import re
# from sklearn.metrics import log_loss
from sklearn.base import clone
from sklearn.utils import check_X_y, check_array
from .multilabel import ProbabilisticClassifierChain

class RuleFitWrapperCV:

    def __init__(self, Cs = [1, 2, 4, 8, 16, 32], cv=10, rank='median', random_state=None, option='pcc'):
        """
        Input: Cs: C candidates list, orginal Cs is [0.1, 0.5, 1, 2, 4, 8, 16, 32]. To save time, we get rid of 0.1 and 0.5
               n_splits: default is 5 Folder cross validation, if n_splits = n, leave one out cross validation, 
                        Choose large number of splits are safety for probabiltic classifier in multi-label classification.
               rank: selecting C criteria, 'median' or 'mean'
        """
        self.cs = Cs
        self.n_splits = cv
        self.rank = []
        self.error = [[] for _ in range(cv)]
        self.rank_option = rank
        self.chain = None
        self.rf = None
        self.num_rules = [[] for _ in range(cv)]
        self.random_state = random_state
        self.option = option
        if cv < 2: raise Exception ('n_splits should at least 2')
        if rank not in ['median', 'mean']: raise Exception ('Invalid ranking method')

    def fit(self, x, y):
        """
        X, y should be dataframe. Apply 2 methods ranking here.
        """
        i = 0
        kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        try:
            x, y = x.values, y.values
        except:
            x, y = x, y
        rf = RuleFit(rfmode='classify', model_type='lr', Cs=[1])
        if self.option == 'pcc':
            self.chain = ProbabilisticClassifierChain(rf)
            self.chain.fit(x,y)
        return self
        # StrtifiedKFold can not splitted multi-dimensions in y. Not useful in this part. We try KFold cv.
        # In this probabilistic classifier, it is dangerous to have small number of data size. For example, if one observation is not found in training samples, it will raise errors.
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if self.option == 'pcc':
                rulefits = [ProbabilisticClassifierChain(RuleFit(rfmode='classify', model_type='lr', Cs=[C], n_jobs=-1)) for C in self.cs]
            else:
                rulefits = [RuleFit(rfmode='classify', model_type='lr', Cs=[C], n_jobs=-1) for C in self.cs]
          
            for each in rulefits:
                each.fit(x_train, y_train)
                y_pred = each.predict_proba(x_test)
                test_error = self.logloss(y_test, y_pred)
                self.error[i].append(test_error)
            self.rank.append([sorted(self.error[i]).index(l) for l in self.error[i]])
            i += 1
        if self.rank_option =='median':
                self.rank = np.median(np.array(self.rank), axis=0)
        else:
            self.rank = np.mean(np.array(self.rank), axis=0)
        lst = list(self.rank)
        indx = lst.index(min(lst))
        rf = RuleFit(rfmode='classify', model_type='lr', Cs=[self.cs[indx]])
        if self.option == 'pcc':
            self.chain = ProbabilisticClassifierChain(clone(rf))
            self.chain.fit(x,y)
        else:
            self.rf = clone(rf)
            self.rf.fit(x, y)
        print(self.cs[indx], 'i99999')
        print(self.rank)
        return self

    def logloss(self, y_test, y_pred):
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
    # def logloss(self, y_test, y_pred):
    #     y_test = y_test.astype(np.float16)
    #     y_pred = y_pred.astype(np.float16)
    #     def _loss(y_test, y_pred):
    #         loss = 0
    #         if not y_pred:
    #             loss -= (1.0-y_test)*np.log2(1.0-y_pred)
    #         elif y_pred == 1:
    #             loss -= y_test*np.log2(y_pred)
    #         else:
    #             loss -= ((1.0-y_test)*np.log2(1.0-y_pred) + y_test*np.log2(y_pred))
    #         return loss
        
    #     if len(y_test.shape) == 1:
    #         N = y_test.shape[0]
    #         loss = 0
    #         for i in range(N):
    #             loss -= _loss(y_test[i], y_pred[i])
    #             loss = loss/N
    #     else:
    #         N,M = y_test.shape
    #         a=[]
    #         for m in range(M):
    #             loss=0
    #             for i in range(N):
    #                 loss -= _loss(y_test[i,m], y_pred[i,m])
    #             loss = loss/N
    #             a.append(round(loss,8))
    #         loss = np.mean(a)
    #     return loss

    def predict_proba(self, x):
        """This is only for predicting probability of p(y=1).
        """
        x = check_array(x)
        if self.chain:
            return self.chain.predict_proba(x)
        return self.rf.predict_proba(x)

    def predict(self, x):
        """This is for converting probability to binary (0, 1)
        """
        x = check_array(x)
        # return [1 if each else 0 for each in self.predict_proba(x)[:,1] > 0.5]
        if self.chain:
            result = self.predict_proba(x)
            result[result>=0.5] = 1
            result[result<0.5] = 0
        else:
            result = self.rf.predict(x)
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

    def get_rules(self, X, y):
        """Get positive rules
        Input: X: features
               y: targets
        """
        y_col = y.columns.tolist()
        sum_columns = X.columns.tolist() + y_col
        res = {}
        rf = [self.model]
        for i, est in enumerate(rf):
            rules = est.get_rules()
            rules = rules[rules['coef'] > 0]
            rules.iloc[np.argsort(rules['importance'])][::-1]
            rules['rule'] = rules.apply(lambda x: self.format_rules(sum_columns, x.rule), axis=1)
            res[y_col[i]] = rules
        return res