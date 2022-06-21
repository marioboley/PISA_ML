"""
This module encapsulates file input/output under the conventation that 
all code is run from the directory of this module as working directory.
"""

import os

PROJECT_ROOT_DIR = "."
DATAPATH = os.path.join(PROJECT_ROOT_DIR, "data")
OUTPUTPATH = os.path.join(PROJECT_ROOT_DIR, "output")

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

