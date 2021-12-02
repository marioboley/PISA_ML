import numpy as np
from copy import deepcopy


def equal_width_1d(data, k, a=None, b=None):
    """Partition numerical data into bins of equal width.
    :param array data: data to be discretized
    :param int k: nuber of bins
    :param int|float a: virtual min value of partioned interval (default data.min())
    :param int|float b: virtual max value of partioned inverval (default data.max())
    :returns: array of dtype int corresponding to binning
    For example:
    >>> equal_width(np.array([0, 0.5, 2, 5, 10]), 10)
    array([0, 0, 1, 4, 9])
    >>> equal_width(np.array([2.0, 3.5, 2.7]), 3)
    array([0, 2, 1])
    >>> equal_width(np.array([2.0, 3.5, 2.7]), 5, a=0, b=5)
    array([1, 3, 2])
    >>> equal_width(np.array([[2.0, 0], [3.5, 0.5], [2.7, 5]]), 3)
    array([[0, 0], [2, 0], [1, 2]])
    >>> equal_width(np.array([[0, 0], [0, 0], [0, 0]]), 3)
    array([[0, 0], [0, 0], [0, 0]])
    """
    res = np.zeros_like(data, dtype='int')
    a = data.min() if a is None else a
    b = data.max() if b is None else b
    h = (b - a) / k
    for i in range(len(data)):
        if h == 0:
            res[i] == 0
        else:
            res[i] = int((data[i] - a) // h) - (data[i] % h == 0 and data[i] != a)
            res[i] = k - 1 if res[i] >= k - 1 else res[i]
    return res


def equal_width(data, k, a=None, b=None):
    res = np.zeros_like(data, dtype='int')
    d = len(data[0]) if type(data[0]) in [np.array, np.ndarray, list] else 1
    for j in range(d):
        if d != 1:
            subdata = data[:, j] if d != 1 else data
            res[:, j] = equal_width_1d(subdata, k=k, a=a, b=b)
        else:
            res = equal_width_1d(data, k=k, a=a, b=b)
    return res


def equal_freq(data, k):
    """Partition numerical data into bins of equal frenquency.
        :param array data: data to be discretized
        :param int k: nuber of bins
        :returns: array of dtype int corresponding to binning
        For example:
        >>> equal_freq(np.array([0, 0.5, 2, 5, 10]), 3)
        array([0, 0, 1, 1, 2])
        >>> equal_freq(np.array([2.0, 3.5, 2.7]), 1)
        array([0, 0, 0])
        >>> equal_freq(np.array([2.0, 3.5, 2.7]), 2)
        array([0, 1, 0])
        >>> equal_freq(np.array([[2.0, 1], [3.5,2], [2.7,3]]), 2)
        array([[0, 0], [1, 0], [0, 1]])
        """
    d = len(data[0]) if type(data[0]) in [np.array, np.ndarray, list] else 1
    new_bins_list = np.zeros_like(data, dtype='int')
    for j in range(d):
        subdata = data[:, j] if d != 1 else data
        bins_dict = dict(zip([_ for _ in range(len(subdata))], subdata))
        bins_list = sorted(bins_dict.items(), key=lambda x: x[1])
        bins_list = [list(each) for each in bins_list]
        for i in range(len(subdata)):
            bins_list[i][1] = i + 1
        length = len(subdata) // k  # each length length + 1
        remain = len(subdata) - k * length
        result_list = [[] for _ in range(k)]
        l = 0
        for each in result_list:
            new_length = length + 1 if remain > 0 else length
            while new_length != 0:
                each.append(bins_list[l][0])
                l += 1
                new_length -= 1
            remain -= 1
        for i in range(len(result_list)):
            for each in result_list[i]:
                if d != 1:
                    new_bins_list[each][j] = i
                else:
                    new_bins_list[each] = i
    return new_bins_list


def refine_subequal_freq(subdata):
    bins_dict = dict(zip([_ for _ in range(len(subdata))], subdata))
    bins_list = sorted(bins_dict.items(), key=lambda x: x[1])
    bins_list = [list(each) for each in bins_list]
    for i in range(len(subdata)):
        bins_list[i][1] = i + 1
    length = len(subdata) // 2  # each length length + 1
    remain = len(subdata) - 2 * length
    result_list = [[] for _ in range(2)]
    l = 0
    for each in result_list:
        new_length = length + 1 if remain > 0 else length
        while new_length != 0:
            each.append(bins_list[l][0])
            l += 1
            new_length -= 1
        remain -= 1
    return result_list


def refine_equal_freq1d(orgdata, modi_data=None):
    """Partition numerical data into bins of equal frenquency.
        :param orgdata: data to be discretized
        :param modi_data: already discretized by this algorithm
        :returns: array of dtype int corresponding to binning
        For example:
        >>> refine_equal_freq1d(np.array([0, 0.5, 2, 5, 10]))
        [[0, 1, 2], [3, 4]], 0
        >>> refine_equal_freq1d(np.array([2.0, 3.5, 2.7]))
        [[0, 2], [1]], 1
        # >>> refine_equal_freq1d(np.array([[2.0, 1], [3.5,2], [2.7,3]]))
        # array([[0, 0], [1, 0], [0, 1]])
        """
    #     d = False if type(data[0]) in [np.array, np.ndarray, list] else True
    poss_out = []
    iden_list = []
    if not modi_data:
        new_list = refine_subequal_freq(orgdata)
        indx_list = builtcount(orgdata, new_list)
        iden_list.append(indx_list)
        return [new_list], iden_list
    current = len(modi_data)
    for k in range(current):
        new_list = deepcopy(modi_data)
        # new_list = modi_data.copy()
        suba = refine_subequal_freq(new_list[k])
        for i in range(len(suba)):
            for j in range(len(suba[i])):
                suba[i][j] = modi_data[k][suba[i][j]]

        indx = k
        for code in suba:
            indx += 1
            new_list.insert(indx, code)
        new_list.pop(k)
        new_list = [each for each in new_list if each]
        if len(new_list) == current + 1:
            poss_out.append(new_list)
            indx_list = builtcount(orgdata, new_list)
            iden_list.append(indx_list)
    return poss_out, iden_list


def builtcount(orgdata, new_list):
    indx_list = deepcopy(orgdata)
    for i in range(len(new_list)):
        for each in new_list[i]:
            indx_list[each] = i
    return indx_list


def getCandiPoint(x, method, max_bins):
    """
    get candidate points, # for multi-dimension data only, ***no duplicated point***
    limitation for this is we can't handle uniform distribution
    return: all candiate subset
    """
    if type(x) in [np.matrix]:
        x = np.array(x).tolist()
        x = [code for each in x for code in each]
    x = np.unique(x)
    current = method(np.array(x), max_bins)  # apply methods
    x = list(x)
    new_x = [[] for _ in range(max(current) + 1)]  # check how many bins here
    for i in range(len(current)):
        indx = current[i]
        new_x[indx].append(x[i])  # real x in different bins
    # get first bins:
    cut_point_lst = []  # store cut point
    for each in new_x:
        cut_point_lst.append(each[0])
        cut_point_lst.append(each[-1]) if len(each) >= 2 else cut_point_lst
    cut_point_lst = [code for code in cut_point_lst if code != min(x) and code!= max(x)]
    return cut_point_lst


# def getCandiSet(x, method, max_bins, n=3):
#     """
#     get candidate points
#     limitation for this is we can't handle uniform distribution
#     return: all candiate subset
#     """
#     if type(x) in [np.matrix]:
#         x = np.array(x).tolist()
#         x = [code for each in x for code in each]
#     current = method(np.array(x), n * max_bins)  # apply methods
#     x = list(x)
#     new_x = [[] for _ in range(max(current) + 1)]  # check how many bins here
#     for i in range(len(current)):
#         indx = current[i]
#         new_x[indx].append(x[i])  # real x in different bins
#     all_subset = [each for each in new_x if each]  # remove empty list
#     return all_subset  # all subset

def cutPointBin(x, known_cut_lst, k):
    """
    Convert selected candidate points to cut point bins
    Input: x: one dimension data
        known_cut_lst: already selected points list
        k: new points
    Output:
        current bin list
    """
    if type(x) in [np.matrix]:
        x = np.array(x).tolist()
        x = [code for each in x for code in each]
    temp = known_cut_lst + [k]
    temp.sort()
    cnt_value = 1
    uniqx = np.unique(x)
    current_list = [None for _ in range(len(uniqx))]
    result_list = [None for _ in range(len(x))]
    for i in range(len(temp) + 1):  # temp index
        for j in range(len(uniqx)):  # x index
            if i == 0:
                if uniqx[j] <= temp[0]:
                    current_list[j] = cnt_value
            elif i == len(temp):
                if uniqx[j] >= temp[-1]:
                    current_list[j] = cnt_value
            else:
                if temp[i - 1] <= uniqx[j] < temp[i]:
                    current_list[j] = cnt_value
        cnt_value += 1
    for i in range(len(uniqx)):
        for j in range(len(x)):
            if uniqx[i] == x[j]:
                result_list[j] = current_list[i]
    return result_list


# def cutPointBin(known_cut_list, candi_subset, k):
#     """
#     This algorithm is convert x to bins index based on subset
#     known_cut_list: index of subset, don't comparison the value
#     candi_subset: all possible subset
#     k: index of candiate subset
#     """
#     if k in known_cut_list:
#         raise Exception(str(k), "already in the list")
#     current_cut = known_cut_list + [k]
#     current_cut.sort() # for safety
#     new_subset = [[] for _ in range(len(current_cut))]
#     if not known_cut_list:
#         for j in range(len(candi_subset)):
#             if j <= k:
#                 new_subset[0] += candi_subset[j]
#             else:
#                 new_subset[1] += candi_subset[j]
#         return new_subset
#     i, j = 0,0
#     while i <= len(current_cut) - 1 and j <= len(candi_subset) -1:
#         if i == 0 and current_cut[i] >= j:
#             new_subset[i] += candi_subset[j]
#             j += 1
#         elif i == len(current_cut) - 1:
#             if current_cut[i] > j >= current_cut[i-1]:
#                 new_subset[i] += candi_subset[j]
#             if current_cut[i] >= j:
#                 new_subset[i] += candi_subset[j]
#             j += 1
#         elif i != len(current_cut) - 1 and current_cut[i + 1] > j >= current_cut[i]:
#             new_subset[i] += candi_subset[j]
#             j += 1
#         else:
#             i += 1
#     return new_subset


if __name__ == '__main__':
    import doctest

    doctest.testmod()
