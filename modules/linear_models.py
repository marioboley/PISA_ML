import statsmodels.api as sm
from copy import deepcopy
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

class GlmWrapper:

    def __init__(self, intercept=True, penalty=True, max_iter=50, alpha=0.1, L1_wt=1) -> None:
        """Default penalty is "elastic net", if L1_wt = 0, ridge penalty
        if L1_wt = 1, lasso penalty
        """
        self.model = None
        self.fitted = None
        self.intercept = intercept
        self.penalty = penalty
        self.L1_wt = L1_wt
        self.maxiter = max_iter
        self.alpha = alpha

    def fit(self, x, y):
        if self.intercept:
            x = sm.add_constant(x, has_constant='add')
        self.model = sm.GLM(y, x, family=sm.families.Binomial())
        self.fitted = self.model.fit_regularized(alpha=self.alpha, maxiter=self.maxiter, L1_wt=self.L1_wt) if self.penalty else self.model.fit()
        return self

    def predict_proba(self, x):
        """This is only for predicting probability of p(y=1).
        """
        if self.intercept:
            x = sm.add_constant(x, has_constant='add')
        return np.column_stack((1-self.fitted.predict(x), self.fitted.predict(x)))

    def predict(self, x):
        """This is for converting probability to binary (0, 1)
        if p>0.5, label=1, otherwise 0
        """
        if self.intercept:
            x = sm.add_constant(x, has_constant='add')
        return np.where(self.fitted.predict(x)>=0.5, 1, 0)

class GlmWrapperCV(GlmWrapper):

    def __init__(self, intercept=True, penalty=True, max_iter=50, alpha=0.1, L1_wt=1, cv=3, Kfold=True):
        GlmWrapper.__init__(self, intercept, penalty, max_iter, alpha, L1_wt)
        self.cv=cv
        self.splitter = KFold(self.cv, shuffle=True, random_state=1000) if Kfold else None

    def fit(self, x, y):
        bestrisk = np.infty
        for train_idx, test_idx in self.splitter.split(x, y, None):
            try:
                x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            except:
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

            model = GlmWrapper(intercept=self.intercept, penalty=self.penalty, max_iter=self.maxiter, alpha=self.alpha, L1_wt=self.L1_wt)
            model.fit(x_train, y_train)
            risk = sum(abs(y_test - model.predict(x_test)))
            if bestrisk > risk:
                self.fitted, bestrisk = model, risk
        return self

class LogisticWrapperCV(LogisticRegression):

    def __init__(self, penalty='l2', *, dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1, class_weight=None, 
                random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None,
                Kfold=True, cv=5):
        super().__init__(penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver, 
                        max_iter, multi_class, verbose, warm_start, n_jobs, l1_ratio)
        self.cv=cv
        self.splitter = KFold(self.cv, shuffle=True, random_state=random_state) if Kfold else None

    def fit(self, x, y):
        bestrisk = np.infty
        for train_idx, test_idx in self.splitter.split(x, y, None):
            print(x, y)
            try:
                x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            except:
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

            model = LogisticRegression(self.penalty, self.dual, self.tol, self.C, self.fit_intercept, self.intercept_scaling, self.class_weight, self.random_state, self.solver, self.max_iter, 
                                        self.multi_class, self.verbose, self.warm_start, self.n_jobs, self.l1_ratio)
            model.fit(x_train, y_train)
            risk = sum(abs(y_test - model.predict(x_test)))
            if bestrisk > risk:
                self.fitted, bestrisk = model, risk