import os
import pickle

from common import OUTPUTPATH, ignore_warning
import data
from models import linear_pcc, gam_pcc, rule_fit_pcc, random_forest_pcc
from modules.experiments import Experiment, KFold, hamming_loss, error, NegLogLikelihoodEvaluator
ignore_warning()

seed=1000
full_estimators = [linear_pcc, gam_pcc, rule_fit_pcc, random_forest_pcc]
full_names = ['LR', 'GAM','RuleFit', 'RF']

interpolation = Experiment(full_estimators, 
                            full_names,
                            KFold(30, shuffle=True, random_state=seed),
                            data.x1, data.y.replace(-1.0, 0.0),
                            groups=data.comp_ids.array,
                            evaluators=[hamming_loss, error, NegLogLikelihoodEvaluator(base=2)],
                            verbose=True).run()

with open(os.path.join(OUTPUTPATH, 'interpolation.pkl'), 'wb') as f:   
    pickle.dump(interpolation, f)
interpolation.summary().to_csv(os.path.join(OUTPUTPATH, 'interpolation.csv'))
