import os
import pickle

import data
from common import OUTPUTPATH, ignore_warning
from modules.experiments import ExtrapolationExperiment, GroupKFoldSpecial, hamming_loss, GroupDescription, error, NegLogLikelihoodEvaluator
from models import linear_pcc, gam_pcc, random_forest_pcc, rule_fit_pcc
ignore_warning()

full_estimators = [linear_pcc, gam_pcc, rule_fit_pcc, random_forest_pcc]
full_names = ['LR', 'GAM','RuleFit', 'RF']

extrapolation = ExtrapolationExperiment(full_estimators, 
                            full_names,
                            GroupKFoldSpecial(len(set(data.comp_ids)), size=22),
                            data.x1, data.y.replace(-1.0, 0.0), groups=data.comp_ids.array,
                            evaluators=[hamming_loss, error, NegLogLikelihoodEvaluator(base=2),
                            GroupDescription(data.comp_descr, 'composition')],
                            verbose=True).run()

with open(os.path.join(OUTPUTPATH, 'extrapolation.pkl'), 'wb') as f:   
    pickle.dump(extrapolation, f)
extrapolation.summary().to_csv(os.path.join(OUTPUTPATH, 'extrapolation.csv'))
            