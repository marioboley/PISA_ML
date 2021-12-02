"""
GAM model for building variables
"""
from sklearn.base import BaseEstimator
from pygam.terms import TermList, te, SplineTerm
from pygam import LinearGAM, LogisticGAM
from functions import *


class VariableStructureGAM:

    def __init__(self, indi=None, multi=None, kind=LogisticGAM):
        self.kind = kind
        self.GAM_ = None
        self.multi_variables = multi
        self.indi_variables = indi

    def gam(self, lam=10, max_iter=1000):
        terms = TermList()
        if self.multi_variables:
            #if self.indi_variables:
            #    for each in self.multi_variables:
            #        for code in each:
            #            if code in self.indi_variables:
            #                self.indi_variables.remove(code)
            # currently don't know how to solve the problems te(1,2,3,4,5), we use hard coding here 2/5
            for each in self.multi_variables:
                if len(each) == 2:
                    terms += te(each[0], each[1], lam=lam)
                elif len(each) == 3:
                    terms += te(each[0], each[1], each[2], lam=lam)
                elif len(each) == 4:
                    terms += te(each[0], each[1], each[2], each[3], lam=lam)
        if self.indi_variables:
            for each in self.indi_variables:
                terms += SplineTerm(each, lam=lam)
        self.GAM_ = self.kind(terms, max_iter=max_iter)
        return self.GAM_

    def fit(self, x, y, lam=10, max_iter=1000):
        self.GAM_ = self.gam(lam=lam, max_iter=max_iter).fit(x, y)
        return self

    def predict(self, x):
        return self.GAM_.predict(x)

    def predict_proba(self, x):
        pass

    def summary(self):
        return self.GAM_, self.multi_variables, self.indi_variables
