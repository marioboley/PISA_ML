"""
Specifies centrally all models used in this study.
"""

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from modules.multilabel import ProbabilisticClassifierChain
from modules.rules import RuleFitWrapperCV
from modules.linear_models import LogisticWrapperCV, GlmWrapper
import numpy as np

STATE = np.random.RandomState(seed=1000)
MAX_ITER = 30000

# Here are the final models
# linear models
linear_base = LogisticRegressionCV(penalty='l2', solver='lbfgs', random_state=STATE, max_iter=MAX_ITER)
linear_pcc = ProbabilisticClassifierChain(linear_base)

# random forest
random_forest_base = RandomForestClassifier(random_state=STATE, min_samples_leaf=1, n_estimators=100)
random_forest_pcc = ProbabilisticClassifierChain(random_forest_base)


# rule fit models
rule_fit_base = RuleFitWrapperCV(Cs = [1, 2, 4, 8, 16, 32], cv=10, rank='median')
rule_fit_pcc = ProbabilisticClassifierChain(RuleFitWrapperCV)

# Here are the NON-USE models
linear_non_base = LogisticRegression(penalty='none', max_iter=MAX_ITER)
glm_non_base = GlmWrapper(intercept=True, penalty=False, max_iter=MAX_ITER)
glm_non_pcc = ProbabilisticClassifierChain(glm_non_base)
linear_non_pcc = ProbabilisticClassifierChain(linear_non_base)

linear_l1_base = LogisticRegressionCV(penalty='l1', solver='saga', random_state=STATE, max_iter=MAX_ITER)
glm_l1_base = GlmWrapper(intercept=True, penalty=True, max_iter=MAX_ITER, alpha=0.1, L1_wt=1)
glm_l2_base = GlmWrapper(intercept=True, penalty=True, max_iter=MAX_ITER, alpha=0.1, L1_wt=0)
linear_l1_pcc = ProbabilisticClassifierChain(linear_l1_base)
glm_l1_pcc = ProbabilisticClassifierChain(glm_l1_base)
glm_l2_pcc = ProbabilisticClassifierChain(glm_l2_base)
