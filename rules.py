from realkd.patch import RuleFit
from sklearn.model_selection import KFold

class RuleFitWrapper:

    def __init__(self, Cs = [1, 2, 4, 8, 16, 32], n_splits=10, rank='median'):
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
        self.model = None
        self.rf = None
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
            X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
            y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values
            if y.shape[1] != 1:
                # if multiple targets, use chain rules
                rulefits = [ProbabilisticClassifierChain(RuleFit(rfmode='classify', model_type='lr', Cs=[C])) for C in self.cs]
            else:
                rulefits = [RuleFit(rfmode='classify',tree_generator='GradientBoostingClassifier', model_type='lr', Cs=[C]) for C in self.cs]
            for each in rulefits:
                each.fit(X_train, y_train)
                test_error = sum(sum(y_test - each.predict(X_test))**2)/len(X_test)
                # SSE
                self.error[i].append(test_error)
            self.rank.append([sorted(self.error[i]).index(l) for l in self.error[i]])
            i += 1
        # print(self.rank)
        # print(self.error)
        if self.rank_option =='median':
                self.rank = np.median(np.array(self.rank), axis=0)
        else:
            self.rank = np.mean(np.array(self.rank), axis=0)
        lst = list(self.rank)   
        indx = lst.index(min(lst))
        self.rf = RuleFit(rfmode='classify', model_type='lr', Cs=[self.cs[indx]])
        if y.shape[1] != 1:
            self.model = ProbabilisticClassifierChain(self.rf)
        else:
            self.model = self.rf
        self.model.fit(X,y)

    def predict_proba(self, X):
        """This is only for predicting probability of p(y=1).
        """
        return self.model.predict_proba(X)

    def predict(self, X):
        """This is for converting probability to binary (0, 1)
        if p>0.5, label=1, otherwise 0
        """
        result = self.predict_proba(X)
        result[result>=0.5] = 1
        result[result<0.5] = 0
        return result

    def get_rules(self):
        """Get positive rules
        """
        rules = self.rf.get_rules()
        rules = rules[rules['coef'] > 0]
        return rules.iloc[np.argsort(rules['importance'])][::-1]
