from random import random
from pygam import LogisticGAM
from sklearn.model_selection import GridSearchCV
import numpy as np


class GamWrapper(LogisticGAM):

    def __init__(self, terms='auto', max_iter=100, tol=0.0001, callbacks=..., fit_intercept=True, verbose=False, **kwargs):
        super().__init__(terms, max_iter, tol, callbacks, fit_intercept, verbose, **kwargs)

    def fit(self, X, y):
        """
        X, y should be dataframe. Apply 2 methods ranking here.
        """
        try: X, y = X.values, y.yalues
        except: pass
        lams = np.random.rand(10, X.shape[1])
        lams = np.exp(lams)
        parameters = {
            'lam': [x for x in lams]
        }
        # if multiple targets, use first lambda values
        logistic_gam = LogisticGAM(fit_intercept=self.fit_intercept, n_splines=self.n_splines, spline_order=self.spline_order)
        self.model = GridSearchCV(logistic_gam, parameters, cv=3, iid=False, return_train_score=True, refit=True, scoring='neg_mean_squared_error')
        self.model.fit(X, y)        
        return self


    # def predict_proba(self, X):
    #     """This is only for predicting probability of p(y=1).
    #     """
    #     return self.model.predict_proba(X)

    # def predict(self, X):
    #     """This is for converting probability to binary (0, 1)
    #     if p>0.5, label=1, otherwise 0
    #     """
    #     result = self.model.predict(X)
    #     return result


# class LogisticWrapperCV(LogisticRegression):

#     def __init__(self, penalty='l2', *, dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1, class_weight=None, 
#                 random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None,
#                 Kfold=True, cv=5):
#         super().__init__(penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver, 
#                         max_iter, multi_class, verbose, warm_start, n_jobs, l1_ratio)
#         self.cv=cv
#         self.splitter = KFold(self.cv, shuffle=True, random_state=random_state) if Kfold else None

#     def fit(self, x, y):
#         bestrisk = np.infty
#         for train_idx, test_idx in self.splitter.split(x, y, None):
#             print(x, y)
#             try:
#                 x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
#                 y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#             except:
#                 x_train, x_test = x[train_idx], x[test_idx]
#                 y_train, y_test = y[train_idx], y[test_idx]

#             model = LogisticRegression(self.penalty, self.dual, self.tol, self.C, self.fit_intercept, self.intercept_scaling, self.class_weight, self.random_state, self.solver, self.max_iter, 
#                                         self.multi_class, self.verbose, self.warm_start, self.n_jobs, self.l1_ratio)
#             model.fit(x_train, y_train)
#             risk = sum(abs(y_test - model.predict(x_test)))
#             if bestrisk > risk:
#                 self.fitted, bestrisk = model, risk