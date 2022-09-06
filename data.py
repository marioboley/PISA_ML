"""
Encapsulates access to data files.
"""

import pandas as pd
import numpy as np
import os

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
              'temp']

abbrev_predictors = ['clogp_cna',
              'mw_cna',
              'apol_cna',
              'mv_cna',
              'psa_cna',
              'dp_cna',
              'mw_tot_cna',
              'mv_tot_cna',
              'clogp_cre',
              'mw_cre',
              'apol_cre',
              'mv_cre',
              'psa_cre',
              'dp_cre',
              'mw_tot_cre',
              'mv_tot_cre',
              'mass_ratio',
              'vol_ratio',
              'conc',
              'ph',
              'salt',
              'charged',
              'temp']

unit = ['n/a',
        'g/mol',
        '$C^2$m/N',
        '$m^3$/mol',
        '$A^2$/mol',
        'count',
        'g/mol',
        '$m^3$/mol',
        'n/a',
        'g/mol',
        '$C^2$m/N',
        '$m^3$/mol',
        '$\AA^2$/mol',
        'count',
        'g/mol',
        '$m^3$/mol',
        'n/a',
        'n/a',
        'wt%',
        'n/a',
        'M',
        'n/a',
        '$\circ$c']

targets = ['sphere',
           'worm',
           'vesicle',
           'other']

prev_comp_id = -1
comp_id = {}
id_comp = {}
unit_comp = dict(zip(abbrev_predictors + targets, unit + ['n/a', 'n/a', 'n/a', 'n/a']))

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

# def monomer_pairs(x, core_comp, corona_comp):
#     for k in range(len(core_comp)):
#         for j in range(len(corona_comp)):
#             core, corona = x[core_comp], x[corona_comp]
#             core, corona = core.astype('float64'), corona.astype('float64')            
#             current = np.outer(core, corona).round()
#             if current[k][j]:
#                 return (core_comp[k], corona_comp[j])
#     return (None, None)

def get_comp_id(x):
    global prev_comp_id, comp_id, id_comp
    # key = tuple(zip(x.index, monomer_pairs(x, core_comp, corona_comp)))
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

# remove non-assembly points
y = polymers.filter(targets, axis=1)
indx = y[(y.sphere == 0) & (y.worm == 0) & (y.vesicle == 0)& (y.other == 0)].index
polymers = polymers.iloc[~polymers.index.isin(indx)]
polymers = polymers.reset_index(drop=True) # add this line to normalized dataframe

# polymers[targets] = polymers[targets].replace(0, -1) # Comment this line to not normalize
comp_ids = polymers.loc[:, corona_comp+core_comp].apply(get_comp_id, axis = 1)

selected_columns = diff(polymers.columns, core_comp + corona_comp + targets + ['Publication DOI',
 'First author', 'cophases', 'no_assem', 'precipitate', 'initiator'])
x = pd.get_dummies(polymers.filter(predictors + core_comp + corona_comp, axis=1))
x1 = pd.get_dummies(polymers.filter(predictors, axis=1))
abbrev_x1 = x1.rename(columns= dict(zip(predictors, abbrev_predictors)))
x2 = pd.get_dummies(polymers.filter(selected_columns, axis=1))
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
