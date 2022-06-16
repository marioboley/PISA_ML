import sys
# sys.path is a list of absolute path strings
sys.path.append('./modules')

import data1 as data
from Experiment import * 

STATE = np.random.RandomState(seed=1000)
kfold = KFold(30, shuffle=True, random_state=STATE)
# x = data.x
# x1 = data.x1
# y = data.y
# target = data.targets
# sphere = data.sphere
# worm = data.worm
# vesicle = data.vesicle
# other = data.other
# comp_ids = data.comp_ids
# polymers = data.polymers
# predictors = data.predictors
# corona_comp = data.corona_comp
# core_comp = data.core_comp

