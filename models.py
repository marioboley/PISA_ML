"""
Specifies centrally all models used in this study.
"""

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from modules.multilabel import ProbabilisticClassifierChain
from modules.rules import RuleFitWrapperCV
import numpy as np

STATE = np.random.RandomState(seed=1000)

linear_base = LogisticRegressionCV(penalty='l2', solver='lbfgs', random_state=STATE)
linear_pcc = ProbabilisticClassifierChain(linear_base) 

random_forest_base = RandomForestClassifier(random_state=STATE, min_samples_leaf=1, n_estimators=100)
random_forest_pcc = ProbabilisticClassifierChain(random_forest_base)

rule_fit_base = RuleFitWrapperCV(Cs = [1, 2, 4, 8, 16, 32], cv=10, rank='median')
rule_fit_pcc = ProbabilisticClassifierChain(RuleFitWrapperCV)
