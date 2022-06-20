"""
Specifies centrally all models used in this study.
"""

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from multilabel import ProbabilisticClassifierChain, BinaryRelevanceClassifier

import numpy as np

STATE = np.random.RandomState(seed=1000)

linear_base = LogisticRegressionCV(penalty='l2', solver='lbfgs', random_state=STATE)
linear_pcc = ProbabilisticClassifierChain(linear_base) 

random_forest_base = RandomForestClassifier(random_state=STATE, min_samples_leaf=1, n_estimators=200)
random_forest_pcc = ProbabilisticClassifierChain(random_forest_base)
random_forest_ind = BinaryRelevanceClassifier(random_forest_base)
