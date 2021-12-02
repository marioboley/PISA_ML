from estimators import *
import numpy as np

def entropy(Y):
    """
    H(Y)
    """
    # unique, count = np.unique(Y.astype('<U22'), return_counts=True, axis=0)
    entro = naive_estimate(Y)
    # prob = count/len(Y)
    # entro = np.sum((-1)*prob*np.log2(prob))
    return abs(entro)


def jEntropy(X, Y):
    """
    H(X,Y)
    """
    XY = np.c_[X, Y]
    return entropy(XY)


def cEntropy(Y, X):
    """
    H(Y|X) = H(Y,X) - H(X)
    """
    return jEntropy(Y, X) - entropy(X)


def gain(X, Y):
    """
    Information Gain, I(X;Y) = H(Y) - H(Y|X)
    """
    return entropy(Y) - cEntropy(Y, X)


def fGain(X, Y):
    """
    Fraction of information, F(X;Y) = I(X;Y)/H(Y)
    """
    return gain(X, Y) / entropy(Y)


def cab(a, b, result=1):
    """
    Function of C^a_b = a*...(a-b+1)/b!
    """
    if b == 0:
        return 1
    if b < 0:
        raise Exception("Expect boundary larger than 0")
    elif a < b:
        return 0
    while b > 0:
        result *= a / b
        a -= 1
        b -= 1
    return int(result)


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
#     new_x = [[] for _ in range(max(current) + 1)] # check how many bins here
#     for i in range(len(current)):
#         indx = current[i]
#         new_x[indx].append(x[i])  # real x in different bins
#     all_subset = [each for each in new_x if each]  # remove empty list
#     return all_subset # all subset
    # # get first bins:
    # cut_point_lst = []  # store cut point
    # for each in new_x:
    #
    #     cut_point_lst.append(each[0])
    #     cut_point_lst.append(each[-1]) if len(each) >= 2 else cut_point_lst
    # return cut_point_lst
