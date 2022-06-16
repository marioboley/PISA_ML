import numpy as np
from pygam import LogisticGAM

LogisticGAM.predict_proba_pygam = LogisticGAM.predict_proba
LogisticGAM.fit_pygam = LogisticGAM.fit

def predict_proba(self, x):
    p1 = self.predict_proba_pygam(x)
    p0 = 1 - p1
    return np.column_stack([p0, p1])

LogisticGAM.predict_proba = predict_proba

def fit(self, x, y):
    self.classes_ = np.array([0, 1])
    return self.fit_pygam(x, y)

LogisticGAM.fit = fit


# utility function for compiling interaction terms
# from pygam.terms import TermList, s

# def term_list(df):
#     terms = TermList()
#     for i in range(len(df.columns)):
#         terms += s(i, lam=1.0)
#     return terms