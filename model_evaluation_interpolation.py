from models import *
import os
import pickle
from modules.experiments import Experiment, KFold, hamming_loss, error, NegLogLikelihoodEvaluator
from common import data

STATE = np.random.RandomState(seed=1000)
full_estimators = [linear_pcc, gam_pcc, rule_fit_pcc, random_forest_pcc]
full_names = ['Lr_pcc', 'GAM_pcc', 'RuFit_pcc', 'RF_pcc']

interpolation = Experiment(full_estimators, 
                            full_names,
                            KFold(30, shuffle=True, random_state=STATE),
                            data.x1, data.y.replace(-1.0, 0.0),
                            groups=data.comp_ids.array,
                            evaluators=[hamming_loss, error, NegLogLikelihoodEvaluator(base=2)],
                            verbose=True).run()

with open(os.path.join(data.OUTPUTPATH, 'interpolation.pkl'), 'wb') as f:   
    pickle.dump(interpolation, f)
interpolation.summary().to_csv(os.path.join(data.OUTPUTPATH, 'interpolation.csv'))
