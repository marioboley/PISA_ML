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
