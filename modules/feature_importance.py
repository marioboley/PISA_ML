import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance

def estimator_feature_importance(est, X):
    feature_importance = np.array(est.feature_importances_)
    sorted_idx = feature_importance.argsort()
    sorted_feature = list(X.columns[sorted_idx])[::-1]
    importance_scores = feature_importance[sorted_idx][::-1]
    return [sorted_feature, importance_scores]



def individual_importance_dataframe(est, X, y, scoring='neg_log_loss', num_importance=None, n_repeats=100, seed=None):
    """
    This algorithm use linear estimators and get the importance variables where p < 0.05
    To easy comparison, we set importance score: 1- p
    """
    r = permutation_importance(est, X, y, n_repeats=n_repeats, scoring=scoring, random_state=seed)
    non_zero_indx = np.where(r.importances_mean > 0)
    indx = r.importances_mean[non_zero_indx].argsort()[::-1]
    important_variables = X.columns[non_zero_indx][indx]
    importance_scores = r.importances_mean[non_zero_indx][indx]
    coef = est.coef_[0][non_zero_indx][indx]
    num_importance = num_importance if num_importance else X.shape[0]
    df = pd.DataFrame({'variables': important_variables[:num_importance], 'coef': coef[:num_importance], 'importance': importance_scores[:num_importance]})
    return df

def linear_importance_rule(est, X, y, scoring='neg_log_loss', num_importance=None, n_repeats=100, seed=None):
    coef = est.coef_[0]
    importance_scores = abs(coef* np.std(X))
    df = pd.DataFrame({'variables': X.columns.tolist(), 'coef': coef, 'importance': importance_scores})
    df = df.sort_values(by='importance', ascending=False)
    df = df.reset_index(drop=True)
    return df

def linear_importance_dataframe(est, X, Y, scoring = 'neg_log_loss', num_importance=None, n_repeats=100, seed=None, typ='permut'):
    """
    Combine all morphologies plots together
    """
    data1 = pd.concat([X, Y], axis=1)
    cnt = None
    for i in range(3):
        col_indx = -4 + i

        estimator, X, y = est[i], data1.iloc[:, :col_indx], Y.iloc[:, i]

        if typ == 'permut':
            temp_df = individual_importance_dataframe(est=estimator, X=X, y=y, scoring=scoring, num_importance=num_importance, n_repeats=n_repeats, seed=seed)
        else:
            temp_df = linear_importance_rule(est=estimator, X=X, y=y, scoring=scoring, num_importance=num_importance, n_repeats=n_repeats, seed=seed)
        if not cnt: 
            df = temp_df
            cnt = 1
        else:
            df = pd.concat([df, temp_df], axis=1)

    upper_columns = ['Sphere', 'Worm', 'Vesicle']
    lower_columns = ['variables', 'coef', 'importance']
    df.columns = pd.MultiIndex.from_product([upper_columns, lower_columns], names=['Phase', 'Property'])
    return df
