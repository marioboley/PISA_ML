"""
Specifies centrally all models used in this study.
"""

from sklearn.ensemble import RandomForestClassifier
from modules.multilabel import ProbabilisticClassifierChain, BinaryRelevanceClassifier
from sklearn.linear_model import LogisticRegressionCV
from modules.rules import RuleFitWrapperCV
from interpret.glassbox import ExplainableBoostingClassifier
import numpy as np
from common import set_seed

seeds = 1000
STATE = np.random.RandomState(seed=seeds)
set_seed(1000)

MAX_ITER = 30000
# logistic models with l2
linear_base = LogisticRegressionCV(penalty='l2', solver='lbfgs', random_state=STATE, max_iter=MAX_ITER)
linear_pcc = ProbabilisticClassifierChain(linear_base) 

# Random Forest
random_forest_base = RandomForestClassifier(random_state=STATE, min_samples_leaf=1, n_estimators=200)
random_forest_pcc = ProbabilisticClassifierChain(random_forest_base)
random_forest_ind = BinaryRelevanceClassifier(random_forest_base)

# GAM models
gam_base = ExplainableBoostingClassifier(interactions=0,random_state=seeds) # not support numpy
gam_pcc = ProbabilisticClassifierChain(gam_base)

# rule fit models
rule_fit_base = RuleFitWrapperCV(Cs = [1, 2, 4, 8, 16, 32], cv=10, rank='median', random_state=set_seed(1000))
rule_fit_pcc = ProbabilisticClassifierChain(rule_fit_base)


