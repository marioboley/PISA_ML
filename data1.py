"""
Encapsulates access to datafile joined_v1.csv.
"""

import os
import pandas as pd

from common import DATAPATH

DATAFILE = os.path.join(DATAPATH, 'joined_v5.csv')

corona_comp = ['corona_GMA',
              'corona_MPC',
              'corona_AcETMAC',
              'corona_MAA',
              'corona_DMA',
              'corona_PEG',
              'corona_CysMA',
              'corona_GluMA',
              'corona_DMAPS',
              'corona_AEMA',
              'corona_QDMAEMA',
              'corona_HPMAm',
              'corona_KSPMA',
              'corona_MAcEPyr',
              'corona_DSDMA']

core_comp = ['core_BzMA',
             'core_DAAM',
             'core_EGEMA',
             'core_GlyMA',
             'core_HBMA',
             'core_HEMA',
             'core_HPMA',
             'core_MEMA',
             'core_PhA',
             'core_TFMA',
             'core_EGDMA',
             'core_cyclic']

predictors = ['clogp_corona',
              'mon_corona_mw',
              'mon_corona_apol',
              'mon_corona_mv',
              'mon_corona_psa',
              'dp_corona',
              'corona_mw_total',
              'corona_mv_total',
              'clogp_core',
              'mon_core_mw',
              'mon_core_apol',
              'mon_core_mv',
              'mon_core_psa',
              'dp_core',
              'core_mw_total',
              'core_mv_total',
              'ratio_mass',
              'ratio_vol',
              'conc',
              'ph',
              'salt',
              'charged',
              'temp'] # remove 

gams = ['corona_mv_total', 
        'corona_mw_total', 
        'dp_corona', 
        'ratio_vol',
        'ratio_mass']#, 
        # 'core_mw_total', 
        # 'core_mv_total', 
        # 'dp_core',
        # 'clogp_core', 
        # 'core_CrippenMR']

targets = ['sphere',
           'worm',
           'vesicle',
           'other']

prev_comp_id = -1
comp_id = {}
id_comp = {}

def get_comp_id(x):
    global prev_comp_id, comp_id, id_comp
    key = tuple(zip(x.index, x.values))
    if key in comp_id:
        return comp_id[key]
    else:
        prev_comp_id += 1
        comp_id[key] = prev_comp_id
        id_comp[prev_comp_id] = key
        return prev_comp_id

def comp_descr(group_id):
    non_zero_elements = list(filter(lambda comp: comp[1]>0, id_comp[group_id]))
    core_elements_as_string = map(
        lambda comp: comp[0].lstrip('core_') + (f'({round(comp[1], 3)})' if comp[1] < 1 else ''), 
        filter(lambda comp: comp[0].startswith('core_'), non_zero_elements))
    corona_elements_as_string = map(
        lambda comp: comp[0].lstrip('corona_') + (f'({round(comp[1], 3)})' if comp[1] < 1 else ''),
        filter(lambda comp: comp[0].startswith('corona_'), non_zero_elements))
    core_string = '+'.join(core_elements_as_string)
    corona_string = '+'.join(corona_elements_as_string)
    return corona_string+'/'+core_string


def diff(lst1, lst2):
    return [each for each in lst1 if each not in lst2]

polymers = pd.read_csv(DATAFILE, index_col=0)
polymers = polymers.reset_index(drop=True) # add this line to normalized dataframe
# polymers[targets] = polymers[targets].replace(0, -1) # Comment this line to not normalize
comp_ids = polymers.loc[:, corona_comp+core_comp].apply(get_comp_id, axis = 1)

selected_columns = diff(polymers.columns, core_comp + corona_comp + targets + ['Publication DOI',
 'First author', 'cophases', 'no_assem', 'precipitate', 'initiator'])
x = pd.get_dummies(polymers.filter(predictors + core_comp + corona_comp, axis=1))
x1 = pd.get_dummies(polymers.filter(predictors, axis=1))
x2 = pd.get_dummies(polymers.filter(selected_columns, axis=1))
x3 = pd.get_dummies(polymers.filter(gams, axis=1))
y = polymers.filter(targets, axis=1)

sphere = polymers['sphere']
worm = polymers['worm']
vesicle = polymers['vesicle']
other = polymers['other']

from copy import deepcopy

def compute_derived_features(df):
    df.core_mw_total = df.dp_core*df.mon_core_mw
    df.core_mv_total = df.dp_core*df.mon_core_mv
    df.corona_mw_total = df.dp_corona * df.mon_corona_mw
    df.corona_mv_total = df.dp_corona * df.mon_corona_mv
    df.ratio_mass = df.corona_mw_total / (df.corona_mw_total + df.core_mw_total)
    df.ratio_vol = df.corona_mv_total / (df.corona_mv_total + df.core_mv_total)
    
def x_grid_data(prototype, xx1, xx2, x1_var='conc', x2_var='dp_core'):
    """
    Computes virtual x data by varying a given prototype point accross two variables.
    """
    df = pd.DataFrame(columns=x.columns)
    m, n = xx1.shape
    df = df.append([deepcopy(prototype) for _ in range(m*n)], ignore_index=True)
    df[x1_var] = xx1.ravel()
    df[x2_var] = xx2.ravel()
    compute_derived_features(df)
    return df 
