"""
Specifies centrally all models used in this study.
"""

from sklearn.ensemble import RandomForestClassifier
from modules.multilabel import ProbabilisticClassifierChain, BinaryRelevanceClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from modules.rules import RuleFitWrapperCV
from modules.linear_models import GlmWrapper
from modules.gam import GamWrapper
from interpret.glassbox import ExplainableBoostingClassifier
import numpy as np

seeds = 1000
STATE = np.random.RandomState(seed=seeds)

MAX_ITER = 30000
linear_base = LogisticRegressionCV(penalty='l2', solver='lbfgs', random_state=STATE, max_iter=MAX_ITER)
linear_pcc = ProbabilisticClassifierChain(linear_base) 

random_forest_base = RandomForestClassifier(random_state=STATE, min_samples_leaf=1, n_estimators=200)
random_forest_pcc = ProbabilisticClassifierChain(random_forest_base)
random_forest_ind = BinaryRelevanceClassifier(random_forest_base)

# Here are the final models
# GAM models
gam_base = ExplainableBoostingClassifier(random_state=seeds) # not support numpy
gam_pcc = ProbabilisticClassifierChain(gam_base)

# rule fit models
rule_fit_base = RuleFitWrapperCV(Cs = [1, 2, 4, 8, 16, 32], cv=10, rank='median', random_state=STATE)
rule_fit_pcc = ProbabilisticClassifierChain(rule_fit_base)


