from pygam import LogisticGAM
from sklearn.model_selection import GridSearchCV
import numpy as np
# import warnings
# warnings.filterwarnings("ignore")

class GamWrapper(LogisticGAM):

    def __init__(self, n_splines=20, spline_order=5) -> None:
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.model = None

    def fit(self, X, y):
        """
        X, y should be dataframe. Apply 2 methods ranking here.
        """
        try:
            X, y = X.values, y.yalues
        except:
            pass
        lams = np.random.rand(10, X.shape[1])
        lams = np.exp(lams)
        parameters = {
            'lam': [x for x in lams]
        }
        # if multiple targets, use first lambda values
        logistic_gam = LogisticGAM(n_splines=self.n_splines, spline_order=self.spline_order)
        self.model = GridSearchCV(logistic_gam, parameters, cv=3, iid=False, return_train_score=True, refit=True, scoring='neg_mean_squared_error')
        self.model.fit(X, y)        
        return self


    # def predict_proba(self, X):
    #     """This is only for predicting probability of p(y=1).
    #     """
    #     return self.model.predict_proba(X.values)

    # def predict(self, X):
    #     """This is for converting probability to binary (0, 1)
    #     if p>0.5, label=1, otherwise 0
    #     """
    #     result = self.model.predict(X.values)
    #     return result