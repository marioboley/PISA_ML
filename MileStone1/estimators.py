import math
import numpy as np

def get_fingers(data):
    """Get counts of appearances
    :param array data: data to be discretized
    :return: array of counts about appearances
    For example:
    >>> get_fingers(np.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4]))
    array([2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> get_fingers(np.array([2.0, 3.5, 2.7]))
    array([3, 0, 0])
    >>> get_fingers(np.array([2.0, 5.0, 3.0, 3.0, 5.0, 3.0]))
    array([1, 1, 1, 0, 0, 0])
    """
    fingerprints = np.zeros(len(data), dtype='int')
    unique, counts = np.unique(data.astype("<U22"), axis=0, return_counts=True)
    for indx in counts:
        fingerprints[indx-1] += 1
    return fingerprints

def naive_estimate(data):
    """Get counts of appearances
    :param array data: data to be discretized
    :return: naive entropy
    For example:
    >>> naive_estimate(np.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4]))
    1.93012496531497
    >>> naive_estimate(np.array([2.0, 3.5, 2.7]))
    1.5849625007211563
    >>> naive_estimate(np.array([2.0, 5.0, 3.0, 3.0, 5.0, 3.0]))
    1.4591479170272448
    """
    finger = get_fingers(data)
    n = len(finger)
    result = 0
    for i in range(len(finger)):
        result += finger[i] * (i + 1) / n * math.log((i + 1) / n, 2)
    return -result

def miller_estimate(data):
    """Get Miller-Madow entropy
    :param data: numpy array data
    :return: Miller-Madow entropy
    For example:
    >>> miller_estimate(np.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4]))
    2.063458298648303
    >>> miller_estimate(np.array([2.0, 3.5, 2.7]))
    1.9182958340544896
    >>> miller_estimate(np.array([2.0, 5.0, 3.0, 3.0, 5.0, 3.0]))
    1.6258145836939115
    """
    finger = get_fingers(data)
    n = len(finger)
    nai = naive_estimate(data)
    result = nai + (sum(finger) - 1) / (2 * n)
    return result

def coverge(data):
    """Get Coverage adjusted entropy estimation
    :param data: numpy array data
    :return: Coverage adjusted entropy estimation
    For example:
    >>> coverge(np.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4]))
    2.25903992879415
    >>> coverge(np.array([2.0, 3.5, 2.7]))
    0
    >>> coverge(np.array([2.0, 5.0, 3.0, 3.0, 5.0, 3.0]))
    1.8139237516537496
    """
    result = 0
    finger = get_fingers(data)
    n = len(finger)
    f1 = finger[0]
    ps = 1 - f1 / n
    if ps == 0:
        return 0
    for i in range(n):
        fi = finger[i]
        current = -fi * ((i + 1) / n) * ps * math.log((i + 1) / n * ps, 2) / (
                1 - (1 - (i + 1) / n * ps) ** n)
        result += current
    return result


def jack(data):
    """Get Jackknifed naive entropy estimation
    :param data: numpy array data
    :return: Jackknifed naive entropy estimation
    For example:
    >>> jack(np.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4]))
    2.2355622825293118
    >>> jack(np.array([2.0, 3.5, 2.7]))
    2.754887502163469
    >>> jack(np.array([2.0, 5.0, 3.0, 3.0, 5.0, 3.0]))
    1.856024112141725
    """
    current = 0
    n = len(data)
    for i in range(n):
        copy_data = np.delete(data, i, 0)
        current_nai = naive_estimate(copy_data)
        current += current_nai
    nai = naive_estimate(data)
    result = n * nai - (n - 1) / n * current
    return result

def check_parameter(data):
    """
    This part is for checking shrinkage lambda
    :param data: numpy array data
    :return: lambda, tk, n

    For example:
    >>> check_parameter(np.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4]))
    (0.423076923076923, 0.2, 15, [0.35384615384615387, 0.23846153846153845, 0.12307692307692307, 0.16153846153846152, 0.12307692307692307], [0.5303466878514332, 0.49317935832222276, 0.3719837308342713, 0.424854293809566, 0.3719837308342713])
    >>> check_parameter(np.array([2.0, 3.5, 2.7]))
    (1, 0.3333333333333333, 3, [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], [0.5283208335737187, 0.5283208335737187, 0.5283208335737187])
    >>> check_parameter(np.array([2.0, 5.0, 3.0, 3.0, 5.0, 3.0]))
    (1, 0.3333333333333333, 6, [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], [0.5283208335737187, 0.5283208335737187, 0.5283208335737187])
    """
    unique, counts = np.unique(data, return_counts=True)
    p = len(unique)
    n = sum(counts)

    var_unbias = 0
    var_mle = 0
    for i in range(len(unique)):
        var_unbias += (1 / p - counts[i] / n) ** 2
        var_mle += (counts[i] / n) ** 2
    _lambda = 1 if n == 1 or var_unbias == 0 else (1 - var_mle) / ((n - 1) * var_unbias)
    _lambda = 1 if _lambda > 1 else _lambda
    _lambda = 0 if _lambda < 0 else _lambda

    result = 0
    shrink_estimator_list = []
    entropy_list = []
    for i in range(len(counts)):
        shrink_estimator = _lambda / p + (1 - _lambda) * counts[i] / n
        result += -shrink_estimator * math.log(shrink_estimator, 2)
        shrink_estimator_list.append(shrink_estimator)
        entropy_list.append(-shrink_estimator * math.log(shrink_estimator, 2))

    return _lambda, 1/p, n, shrink_estimator_list, entropy_list

def James_estimate(data):
    """Get James Shrinkage entropy estimatation
    :param data: numpy array data
    :return: \hat H^{shrink}
    """
    unique, counts = np.unique(data, return_counts=True)
    p = len(unique)
    n = sum(counts)

    var_unbias = 0
    var_mle = 0
    for i in range(len(unique)):
        var_unbias += (1 / p - counts[i] / n) ** 2
        var_mle += (counts[i] / n) ** 2
    _lambda = 1 if var_unbias == 0 else (1 - var_mle) / ((n - 1) * var_unbias)
    _lambda = 1 if _lambda > 1 else _lambda
    _lambda = 0 if _lambda < 0 else _lambda

    result = 0
    for i in range(len(counts)):
        shrink_estimator = _lambda / p + (1 - _lambda) * counts[i] / n
        result += -shrink_estimator * math.log(shrink_estimator, 2)
    return result


if __name__=='__main__':
    import doctest
    doctest.testmod()