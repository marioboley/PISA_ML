import numpy as np

from sklearn.base import clone
from sklearn.multioutput import MultiOutputClassifier as BinaryRelevanceClassifier
from sklearn.utils import check_X_y, check_array

from itertools import product


class ProbabilisticClassifierChain:

    def __init__(self, baselearner):
        self.baselearner = baselearner
        self.r_ = None
        self.fitted_ = None
        self.patterns_ = None

    def fit(self, x, y):
        x, y = check_X_y(x, y, multi_output=True)
        _, r = y.shape
        self.r_ = r
        self.fitted_ = []
        self.patterns_ = list(np.array(p) for p in product(*(self.r_*[range(2)])))
        for i in range(r):
            _x = np.column_stack([x]+[y[:, :i]])
            _y = y[:, i]
            self.fitted_.append(clone(self.baselearner, safe=False))
            self.fitted_[-1].fit(_x, _y)
        return self

    def predict_proba_of(self, x, y):
        """Predicts probabilities of full phase binary indicator vectors for each row of 
        covariate matrix.

        :param x: np.array|pd.DataFrame: matrix of covariates of shape (n, d)
        :param y: np.array of shape (r, ) or (n, r)
        :return: np.array of shape (n,) with probabilities of provided target vector value(s)
        """
        _x = x.copy()
        if len(y.shape)==1:
            y = y.reshape(1, self.r_)
        idx = np.arange(len(x))
        res = self.fitted_[0].predict_proba(_x)[idx, y[:, 0]]
            
        for i in range(1, self.r_):
            # TODO: avoid reconstruction of full augmented data matrix if possible in np
            _x = np.column_stack([_x]+[np.ones(len(x))*y[:, i-1]])
            try:
                res *= self.fitted_[i].predict_proba(_x)[idx, y[:, i]]
            except:
                res *= self.fitted_[i].predict_proba(_x)
        return res

    def predict_proba(self, x):
        x = check_array(x)
        res = np.column_stack([np.zeros(len(x), dtype=np.float64) for _ in range(self.r_)])
        for y in self.patterns_:
            p = self.predict_proba_of(x, y)
            for i in range(self.r_):
                if y[i]:
                    res[:, i] += p
        return res

    def predict_full_proba(self, x):
        x = check_array(x)
        return np.column_stack([self.predict_proba_of(x, p) for p in self.patterns_])

    def predict(self, x):
        x = check_array(x)
        max_ap = np.argmax(np.column_stack([self.predict_proba_of(x, p) for p in self.patterns_]), axis=1)
        return np.array([self.patterns_[max_ap[i]] for i in range(len(max_ap))]) #how to do this directly as np op?

    def conditional_entropy(self, x):
        pass

BinaryRelevanceClassifier.predict_proba_sklearn = BinaryRelevanceClassifier.predict_proba

def marg_proba_from_list_of_binary(self, x):
    r = len(self.estimators_)
    return np.column_stack([self.predict_proba_sklearn(x)[i][:, 1] for i in range(r)])

BinaryRelevanceClassifier.predict_proba = marg_proba_from_list_of_binary

def predict_proba_of_from_marginals(est, x, y):
    p = est.predict_proba(x)
    return np.multiply.reduce(p*y + (1-p)*(1-y), axis=1)
