from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pandas import DataFrame, Series

import numpy as np
from matplotlib import pyplot as plt

class Experiment:
    """
    Experiment that fits range of estimators across a number of splits

    ver 1
    """

    def __init__(self, estimators, estimator_names, splitter, x, y, groups=None, score=r2_score, verbose=True):
        self.x = x
        self.y = y
        self.groups = groups
        self.estimators = estimators
        self.estimator_names = estimator_names
        self.splitter = splitter
        self.score = score
        self.verbose = verbose
        self.num_reps = self.splitter.get_n_splits(self.x, self.y)
        self.results_ = None
        self.fitted_ = None

    def run(self):
        if self.verbose:
            title = f'Running experiment with {self.splitter.get_n_splits(self.x, self.y, self.groups)} repetitions'
            print(title)
            print('='*len(title))
        self.results_ = DataFrame(columns=['split', 'estimator', 'train_score', 'test_score'])
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
            #train_test_split(self.x, self.y, test_size=self.test_size)
            x_train, x_test = self.x.iloc[train_idx, :], self.x.iloc[test_idx, :]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
            for j, est in enumerate(self.estimators):
                name = self.estimator_names[j]
                if self.verbose > 1:
                    print(name)
                    print('-'*len(name), flush=True)
                _est = est() if callable(est) else clone(est, safe=False)
                _est.fit(x_train, y_train)
                train_score = self.score(y_train, _est.predict(x_train))
                test_score = self.score(y_test, _est.predict(x_test))
                if self.verbose > 1:
                    print(f'train/test R2: {train_score:.3f}/{test_score:.3f}')
                    print()
                self.fitted_[name].append(_est)
                self.results_ = self.results_.append(
                    {
                        'split': i,
                        'estimator': name,
                        'train_score': train_score,
                        'test_score': test_score
                    },
                    ignore_index=True
                )
            if self.verbose==1:
                print('*', end='')
        if self.verbose:
            print()
        return self

    def summary(self):
        res = DataFrame(columns=['mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score'])
        for name in self.estimator_names:
            res = res.append(Series({
                'mean_train_score': self.results_[self.results_['estimator']==name]['train_score'].mean(),
                'std_train_score': self.results_[self.results_['estimator']==name]['train_score'].std(),
                'mean_test_score': self.results_[self.results_['estimator']==name]['test_score'].mean(),
                'std_test_score': self.results_[self.results_['estimator']==name]['test_score'].std()
            }, name=name))
        return res

    def plot_summary(self):
        width = 0.35
        summ = self.summary()
        ind = np.arange(len(summ))
        plt.bar(ind-width/2, summ['mean_train_score'], width=width, label='train', 
                yerr=summ['std_train_score']/self.num_reps**0.5, capsize=3.0)
        plt.bar(ind+width/2, summ['mean_test_score'], width=width, label='test',
                yerr=summ['std_test_score']/self.num_reps**0.5, capsize=3.0)
        plt.ylabel('score')
        plt.legend()
        plt.xticks(ind, summ.index)