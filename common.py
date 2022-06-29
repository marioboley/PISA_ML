"""
This module encapsulates file input/output under the conventation that 
all code is run from the directory of this module as working directory.
"""

import os

PROJECT_ROOT_DIR = "."
DATAPATH = os.path.join(PROJECT_ROOT_DIR, "data")
OUTPUTPATH = os.path.join(PROJECT_ROOT_DIR, "output")
