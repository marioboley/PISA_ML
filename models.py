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
# gam_base = GamWrapper(n_splines=20, spline_order=5, max_iter=MAX_ITER) # these gam models are not working
# gam_pcc = ProbabilisticClassifierChain(gam_base)
gam_base = ExplainableBoostingClassifier(random_state=seeds) # not support numpy
gam_pcc = ProbabilisticClassifierChain(gam_base)

# rule fit models
rule_fit_base = RuleFitWrapperCV(Cs = [1, 2, 4, 8, 16, 32], cv=5, rank='median', random_state=STATE)
rule_fit_pcc = ProbabilisticClassifierChain(rule_fit_base)

rule_fit_base_mean = RuleFitWrapperCV(Cs = [1, 2, 4, 8, 16, 32], cv=5, rank='mean', random_state=STATE)
rule_fit_pcc_mean = ProbabilisticClassifierChain(rule_fit_base_mean)

# Here are the NON-USE models
linear_non_base = LogisticRegression(penalty='none', max_iter=MAX_ITER)
glm_non_base = GlmWrapper(intercept=True, penalty=False, max_iter=MAX_ITER)
glm_non_pcc = ProbabilisticClassifierChain(glm_non_base)
linear_non_pcc = ProbabilisticClassifierChain(linear_non_base)

linear_l1_base = LogisticRegressionCV(penalty='l1', solver='saga', random_state=STATE, max_iter=MAX_ITER)
glm_l1_base = GlmWrapper(intercept=True, penalty=True, max_iter=MAX_ITER, alpha=5, L1_wt=1)
glm_l2_base = GlmWrapper(intercept=True, penalty=True, max_iter=MAX_ITER, alpha=5, L1_wt=0)
linear_l1_pcc = ProbabilisticClassifierChain(linear_l1_base)
glm_l1_pcc = ProbabilisticClassifierChain(glm_l1_base)
glm_l2_pcc = ProbabilisticClassifierChain(glm_l2_base)
