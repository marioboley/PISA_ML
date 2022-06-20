from models import *
from common import *
import os
import pickle

STATE = np.random.RandomState(seed=1000)
full_estimators = [linear_pcc, rule_fit_pcc, random_forest_pcc]
full_names = ['Lr_pcc', 'RuFit_pcc', 'RF_pcc']

extrapolation = ExtrapolationExperiment(full_estimators, 
                            full_names,
                            GroupKFoldSpecial(len(set(data.comp_ids)), size=20),
                            data.x1, data.y.replace(-1.0, 0.0), groups=data.comp_ids.array,
                            evaluators=[hamming_loss, error, NegLogLikelihoodEvaluator(base=2),
                            GroupDescription(data.comp_descr, 'composition')],
                            verbose=True).run()

with open(os.path.join(data.OUTPUTPATH, 'extrapolation.pkl'), 'wb') as f:   
    pickle.dump(extrapolation, f)
extrapolation.summary().to_csv(os.path.join(data.OUTPUTPATH, 'extrapolation.csv'))