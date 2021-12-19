"""
This algorithm is targeted as permutation model i.e expectation of reliable mutual information
"""
from estimators import *
import math
import numpy as np


class Permutation():

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.mi = 0

    def summary(self):
        a_list, ai_list = np.unique(self.x.astype("<U22"), axis=0, return_counts=True)
        b_list, bj_list = np.unique(self.y.astype("<U22"), axis=0, return_counts=True)
        #new_ai_list = [ai_list[list(a_list).index(str(each))] for each in self.x]
        #new_bj_list = [bj_list[list(b_list).index(list(each))] for each in self.y]
        n = len(self.x)
        for ai in ai_list:
            for bj in bj_list:
                #n = min(ai, bj)
                k_min, k_max = max(1, ai + bj - n), min(ai, bj)
                for k in range(k_min, k_max+1):
                    p_k = cab(bj, k) * cab(n-bj, ai-k) / cab(n, ai)
                    self.mi += p_k * k / n * math.log2(k * n / (ai * bj))
        return self.mi
