"""
Specifies centrally all models used in this study.
"""

from sklearn.ensemble import RandomForestClassifier
from modules.multilabel import ProbabilisticClassifierChain, BinaryRelevanceClassifier
from sklearn.linear_model import LogisticRegressionCV
from modules.rules import RuleFitWrapperCV
from interpret.glassbox import ExplainableBoostingClassifier

seeds = 1000

MAX_ITER = 30000
# logistic models with l2
linear_base = LogisticRegressionCV(penalty='l2', solver='lbfgs', random_state=seeds, max_iter=MAX_ITER)
linear_pcc = ProbabilisticClassifierChain(linear_base) 

# Random Forest
random_forest_base = RandomForestClassifier(random_state=seeds, min_samples_leaf=1, n_estimators=200)
random_forest_pcc = ProbabilisticClassifierChain(random_forest_base)
random_forest_ind = BinaryRelevanceClassifier(random_forest_base)

# GAM models
gam_base = ExplainableBoostingClassifier(interactions=0,random_state=seeds)
gam_pcc = ProbabilisticClassifierChain(gam_base)

# rule fit models
rule_fit_base = RuleFitWrapperCV(Cs = [1, 2, 4, 8, 16, 32], cv=10, rank='median', random_state=seeds, 
                                tree_generator=None, exp_rand_tree_size=False)
rule_fit_pcc = ProbabilisticClassifierChain(rule_fit_base)


