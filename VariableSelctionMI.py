"""
This algorithm is for MAIN variable selection
"""
from VariableMI import *
import itertools
from functions import *
import numpy as np


class VariableSelectionMI:

    def __init__(self, x, y, column_names, estimator=naive_estimate, xbin=False, ybin=False, bin_type='freq', x_bin=2,
                 y_bin=2, permut=False, fraction=True, individual=True, interaction=False, combination=2, greedy=False,
                 greedy_variables_initial=False, greedy_top=20):
        self.estimator = estimator
        self.x = x
        self.y = y
        self.bin_type = bin_type
        self.xbin = xbin
        self.ybin = ybin
        self.names = column_names
        self.interaction = interaction
        self.individual = individual
        self.x_bin = x_bin
        self.y_bin = y_bin
        self.mi_list = []
        self.inter_list = []
        self.indi_variables = None
        self.inter_variables = None
        self.permut = permut
        self.fraction = fraction
        self.comb = combination
        self.greedy = greedy
        self.greedy_variables_initial = greedy_variables_initial
        self.greedy_top = greedy_top

    def summary(self):
        if self.individual or self.greedy:
            for i in range(len(self.names)):
                if len(self.names) != 1:
                    mi = VariableMI(np.array(self.x[:, i], dtype=float), self.y, estimator=self.estimator,
                                    xbin=self.xbin,
                                    ybin=self.ybin, bin_type=self.bin_type, x_bin=self.x_bin, y_bin=self.y_bin,
                                    permut=self.permut, fraction=self.fraction).summary()
                else:
                    mi = VariableMI(np.array(self.x, dtype=float), self.y, estimator=self.estimator, xbin=self.xbin,
                                    ybin=self.ybin, bin_type=self.bin_type, x_bin=self.x_bin, y_bin=self.y_bin,
                                    permut=self.permut, fraction=self.fraction).summary()
                self.mi_list.append((mi, self.names[i]))
            self.mi_list.sort(reverse=True)

        if self.greedy:
            index_list = [self.names.index(self.mi_list[0][1])] if not self.greedy_variables_initial else self.greedy_variables_initial
            self.greedy_top = self.greedy_top - 1 \
                if not self.greedy_variables_initial \
                else self.greedy_top - len(self.greedy_variables_initial)
            return_list = [self.mi_list[0][0], self.names.index(self.mi_list[0][1])]
            while self.greedy_top:
                sub_inter_list = []
                for indx in range(len(self.names)):
                    if indx in index_list:
                        continue
                    else:
                        sublist = index_list + [indx]
                        int_mi = VariableMI(np.array(self.x[:, sublist], dtype=float), self.y, estimator=self.estimator,
                                            xbin=self.xbin, ybin=self.ybin, bin_type=self.bin_type, x_bin=self.x_bin,
                                            y_bin=self.y_bin, permut=self.permut, fraction=self.fraction).summary()
                        sub_inter_list.append((int_mi, sublist))
                sub_inter_list.sort(reverse=True)
                index_list = sub_inter_list[0][1]
                self.greedy_top -= 1
                return_list.append([sub_inter_list, [self.names[each] for each in index_list]])
            return return_list

        if self.interaction and not self.greedy:
            all_interactions = list(itertools.combinations([_ for _ in range(len(self.x[0]))], self.comb))
            for cell in all_interactions:
                int_mi = VariableMI(np.array(self.x[:, cell], dtype=float), self.y, estimator=self.estimator,
                                    xbin=self.xbin, ybin=self.ybin, bin_type=self.bin_type, x_bin=self.x_bin,
                                    y_bin=self.y_bin, permut=self.permut, fraction=self.fraction).summary()
                self.inter_list.append((int_mi, (self.names[cell[0]], self.names[cell[1]])))
            self.inter_list.sort(reverse=True)
        else:
            pass

        return self.mi_list, self.inter_list
