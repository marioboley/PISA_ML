from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import get_scorer as get_sklearn_scorer
from sklearn.model_selection import train_test_split, GroupKFold
from pandas import DataFrame, Series
from pandas.api.types import is_numeric_dtype

import numpy as np
from matplotlib import pyplot as plt

import os

PROJECT_ROOT_DIR = "."
DATAPATH = os.path.join(PROJECT_ROOT_DIR, "data")
OUTPUTPATH = os.path.join(PROJECT_ROOT_DIR, "output")

class Evaluator:
    pass

    def applicable_to_test(self):
        return True

    def applicable_to_train(self):
        return True


class SklearnScoreEvaluator(Evaluator):
    """
    >>> acc = SklearnScoreEvaluator('accuracy')
    >>> str(acc)
    'accuracy'
    """

    def __init__(self, scorer):
        self.scorer = get_sklearn_scorer(scorer)

    def __call__(self, est, x, y, groups=None):
        return self.scorer(est, x, y)

    def __str__(self):
        return self.scorer._score_func.__name__.rstrip('_score')


class SampleSize(Evaluator):

    def __call__(self, est, x, y, groups=None):
        return len(x)

    def __str__(self):
        return 'size'

sample_size = SampleSize()

class GroupDescription(Evaluator):

    def __init__(self, id_to_description=lambda gid: gid, name=None, apply_to_train=False):
        self.id_to_description = id_to_description
        self.name = 'group' if name is None else name
        self.apply_to_train = apply_to_train

    def __call__(self, est, x, y, groups):
        return self.id_to_description(groups[0])

    def __str__(self):
        return self.name

    def applicable_to_train(self):
        return self.apply_to_train

class Experiment:
    """
    Experiment that fits range of estimators across a number of splits

    ver 2
    """

    def __init__(self, estimators, estimator_names, splitter, x, y, groups=None, evaluators=['accuracy'], verbose=True):
        self.x = x
        self.y = y
        self.groups = groups
        self.estimators = estimators
        self.estimator_names = estimator_names
        self.splitter = splitter
        self.evaluators = [e if isinstance(e, Evaluator) else SklearnScoreEvaluator(e) for e in evaluators]
        self.verbose = verbose
        self.num_reps = self.splitter.get_n_splits(self.x, self.y)
        self.results_ = None
        self.fitted_ = None

    def run(self):
        if self.verbose:
            title = f'Running experiment with {self.splitter.get_n_splits(self.x, self.y, self.groups)} repetitions'
            print(title)
            print('='*len(title))

        res_cols = ['split', 'estimator']
        for e in self.evaluators:
            if e.applicable_to_train(): res_cols.append(f'train_{e}')
            if e.applicable_to_test(): res_cols.append(f'test_{e}')
        self.results_ = DataFrame(columns=res_cols)
        self.fitted_ = {}
        for name in self.estimator_names:
            self.fitted_[name] = []
        i = -1
        for train_idx, test_idx in self.splitter.split(self.x, self.y, self.groups):
            i += 1
            if self.verbose > 1:
                print('Split', i)
                print('-------')
                print()

            x_train, x_test = self.x.iloc[train_idx, :], self.x.iloc[test_idx, :]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
            groups_train, groups_test = (None, None) if self.groups is None else (self.groups[train_idx], self.groups[test_idx])

            for j, est in enumerate(self.estimators):
                name = self.estimator_names[j]
                if self.verbose > 1:
                    print(name)
                    print('-'*len(name), flush=True)
                _est = est() if callable(est) else clone(est, safe=False)
                _est.fit(x_train, y_train)

                conf_results = {
                    'split': i,
                    'estimator': name,                    
                }
                for e in self.evaluators:
                    if e.applicable_to_train():
                        train_e = e(_est, x_train, y_train, groups_train)
                        conf_results[f'train_{e}'] = train_e
                    if e.applicable_to_test():
                        test_e = e(_est, x_test, y_test, groups_test)
                        conf_results[f'test_{e}'] = test_e
                    if self.verbose > 1:
                        if e.applicable_to_test() and e.applicable_to_train():
                            print(f'train/test {e}: {train_e:.3f}/{test_e:.3f}')
                            print()
                        elif e.applicable_to_test():
                            print(f'test {e}: {test_e:.3f}')
                            print()
                self.fitted_[name].append(_est)
                self.results_ = self.results_.append(conf_results, ignore_index=True)

            if self.verbose==1:
                print('*', end='')
        if self.verbose:
            print()
        self.results_ = self.results_.convert_dtypes()
        return self

    def summary(self):
        res_cols = []
        num_evals = [e for e in self.evaluators 
                             if e.applicable_to_train() and is_numeric_dtype(self.results_[f'train_{e}'])
                             or e.applicable_to_test() and is_numeric_dtype(self.results_[f'test_{e}'])]
        for e in num_evals:
            res_cols.append(f'mean_train_{e}')
            res_cols.append(f'std_train_{e}')
            res_cols.append(f'mean_test_{e}')
            res_cols.append(f'std_test_{e}')
        res = DataFrame(columns=res_cols)
        for name in self.estimator_names:
            est_res = {}
            for e in num_evals:
                est_res[f'mean_train_{e}'] = self.results_[self.results_['estimator']==name][f'train_{e}'].mean()
                est_res[f'std_train_{e}'] = self.results_[self.results_['estimator']==name][f'train_{e}'].std()
                est_res[f'mean_test_{e}'] = self.results_[self.results_['estimator']==name][f'test_{e}'].mean()
                est_res[f'std_test_{e}'] = self.results_[self.results_['estimator']==name][f'test_{e}'].std()
            res = res.append(Series(est_res, name=name))
        return res

    def plot_summary(self, metric):
        width = 0.35
        summ = self.summary()
        ind = np.arange(len(summ))
        plt.bar(ind-width/2, summ[f'mean_train_{metric}'], width=width, label='train', 
                yerr=summ[f'std_train_{metric}']/self.num_reps**0.5, capsize=3.0)
        plt.bar(ind+width/2, summ[f'mean_test_{metric}'], width=width, label='test',
                yerr=summ[f'std_test_{metric}']/self.num_reps**0.5, capsize=3.0)
        plt.ylabel(metric)
        plt.legend()
        plt.xticks(ind, summ.index)


# class TestComposition(Evaluator):

#     def __call__(self, est, x, y, groups):
#         return get_comp_id(x)


class ExtrapolationExperiment(Experiment):

    def __init__(self, estimators, estimator_names, x, y, groups, score=['accuracy', sample_size], verbose=True):
        Experiment.__init__(self, estimators, estimator_names, GroupKFold(len(set(groups))), x, y, groups, score, verbose)


if __name__=='__main__':
    from doctest import testmod
    testmod()