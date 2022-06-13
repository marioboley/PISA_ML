import statsmodels.api as sm
from copy import deepcopy
import numpy as np

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
            x = sm.add_constant(x)
        self.model = sm.GLM(y, x, family=sm.families.Binomial())
        self.fitted = self.model.fit_regularized(alpha=self.alpha, maxiter=self.maxiter, L1_wt=self.L1_wt) if self.penalty else self.model.fit()
        return self.fitted

    def predict_proba(self, x):
        """This is only for predicting probability of p(y=1).
        """
        if self.intercept:
            x = sm.add_constant(x)
        return np.column_stack((1-self.fitted.predict(x), self.fitted.predict(x)))
    

    def predict(self, x):
        """This is for converting probability to binary (0, 1)
        if p>0.5, label=1, otherwise 0
        """
        if self.intercept:
            x = sm.add_constant(x)
        return  np.where(self.fitted.predict(x)>=0.5, 1, 0)
    
    # def __call__(self):
    #     return deepcopy(self)

    # def __call__(self, *args, **kwds):
    #     pass

    # def get_parms(self):
    #     pass

    # def set_params(self):
    #     pass