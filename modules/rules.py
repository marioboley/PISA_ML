from realkd.patch import RuleFit
from sklearn.model_selection import KFold
import numpy as np
import re
from sklearn.metrics import log_loss
from sklearn.utils import check_X_y, check_array

class RuleFitWrapperCV:

    def __init__(self, Cs = [1, 2, 4, 8, 16, 32], cv=10, rank='median', random_state=None):
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
        self.model = None
        self.num_rules = [[] for _ in range(cv)]
        self.random_state = random_state
        if cv < 2: raise Exception ('n_splits should at least 2')
        if rank not in ['median', 'mean']: raise Exception ('Invalid ranking method')

    def fit(self, x, y):
        """
        X, y should be dataframe. Apply 2 methods ranking here.
        """
        i = 0
        kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        x, y = check_X_y(x, y)
        if len(self.cs) == 1:
            self.model = RuleFit(rfmode='classify', model_type='rl', Cs=self.cs)
            self.model.fit(x, y)
            return self
        # StrtifiedKFold can not splitted multi-dimensions in y. Not useful in this part. We try KFold cv.
        # In this probabilistic classifier, it is dangerous to have small number of data size. For example, if one observation is not found in training samples, it will raise errors.
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            rulefits = [RuleFit(rfmode='classify', model_type='rl', Cs=[C]) for C in self.cs]  
          
            for each in rulefits:
                each.fit(x_train, y_train)
                y_pred = each.predict_proba(x_test)
                test_logloss = log_loss(y_test, y_pred, eps=1e-15, normalize=True, sample_weight=None, labels=[1,0])
                self.error[i].append(test_logloss)
            self.rank.append([sorted(self.error[i]).index(l) for l in self.error[i]])
            i += 1
        if self.rank_option =='median':
            self.rank = np.median(np.array(self.rank), axis=0)
        else:
            self.rank = np.mean(np.array(self.rank), axis=0)
        lst = list(self.rank)
        indx = lst.index(min(lst))
        self.model = RuleFit(rfmode='classify', model_type='rl', Cs=[self.cs[indx]])
        self.model.fit(x, y)
        return self

    def predict_proba(self, x):
        """This is only for predicting probability of p(y=1).
        """
        x = check_array(x)
        return self.model.predict_proba(x)

    def predict(self, x):
        """This is for converting probability to binary (0, 1)
        """
        x = check_array(x)
        return  self.model.predict(x)

    def format_rules(self, feature_names, data, types='rule'):
        """This is for format the rule features
        Input: feature_names: feature names
                data: dataframe
        """
        dic_rules = dict(zip(['feature_' + str(i) + ' ' for i in range(len(feature_names))], feature_names))
        for key in dic_rules:
            if types == 'rule':
                if re.findall(r'(' + key +')', data):
                    data = re.sub(key, dic_rules[key], data)
                temp_data = data.split()
                for i in range(len(temp_data)):
                    try: temp_data[i] = str(round(float(temp_data[i]),4))
                    except: pass
                data = " ".join(temp_data)
            else:
                if key[:-1] == data:
                    data = dic_rules[key]
        return data

    def get_rules(self, columns):
        """Get positive rules
        Input: X: features
               y: targets
        """
        rules = self.model.get_rules()
        rules = rules[rules['importance'] != 0]
        rules = rules.iloc[np.argsort(rules['importance'])][::-1]
        rules['rule'] = rules.apply(lambda x: self.format_rules(columns, x.rule, x.type), axis=1)
        return rules

        