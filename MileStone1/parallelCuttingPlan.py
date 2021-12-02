from binning import *
from functions import *
from itertools import combinations
from permutation import *


def reliable_MI1d(x, y):
    """
    calculate reliable fraction of mi
    """
    return (gain(np.array(x), np.array(y)) - Permutation(np.array(x), np.array(y)).summary()) / entropy(
        np.array(y))


class Cutplan2d():

    def __init__(self, x, y, candi_list):
        """
        Only for two dimensional data
        This algorithm is designed for refinement of any participation method,
        we first cut these points as candidate points. i.e. use equal frequency here
        After that, we greedy select these candidate points and choose highest reliable MI
        Input: x: predictors
               y: target variables
               un_points: int, number of unselected points
               _bins: list, current bins performance
               candi_list: list, all candidate points
               known_cutpoint: list already selected cut points.
        Output: known_cut_lst: all selected cut points values list if bin=5, only 4 values here
                candi_mi: maximum reliable fraction of MI
                candi_list: current bin list looks like
        """
        self.x = x
        self.y = y
        self.candi_list = candi_list
        self.mi = 0
        self.relat_list = [] # record related dimensions
        self.result = {}

    def __call__(self):
        # using candidate points do binning.
        self.relat_list = self.get_related()
        for i in range(len(self.relat_list)):
            print("i:", i)
            indx1, indx2 = self.relat_list[i]
            key = "/".join([str(indx1), str(indx2)])
            if key not in self.result.keys():
                self.result[key] = []
            cut1, cut2 = self.candi_list
            known1, known2 = [], []
            candi_mi = 0
            # only for two dimensions
            # while not # control how many cuts
            cnt = 0
            for each in cut1:
                print(cnt, len(cut1) * len(cut2))
                if each not in known1:
                    current1 = cutPointBin(self.x[:, indx1], known1, each)
                    for code in cut2:
                        if code not in known2:
                            current2 = cutPointBin(self.x[:, indx2], known2, code)
                            new_x = np.column_stack((current1, current2))
                            current_mi = reliable_MI1d(new_x, self.y)
                            if current_mi > candi_mi:
                                candi_point_list1 = known1 + [each]
                                candi_point_list2 = known2 + [code]
                                candi_mi = current_mi
                cnt += 1
            known2 = candi_point_list2
            known1 = candi_point_list1
            self.result[str(indx1) + "/" + str(indx2)].append([candi_mi, known1, known2])
        return self.result


    def get_related(self):
        # check every two dimension data.
        # if two dimension data mi <= 0.05, ignore, if mi >= 0.05 accept
        # filter relevant dimension

        #comb_list = [each for each in combinations(range(len(self.x[0].A1)), 2)]
        # for i in range(len(comb_list)):
        #     indx_mi = reliable_MI1d(self.x[:, comb_list[i]], self.y)
        #     if indx_mi > 0.05:
        #         self.relat_list.append(comb_list[i])
        return comb_list

    def __str__(self):
        print('related dimensions are: ', self.relat_list)
        pass