import numpy as np

from sklearn.inspection import plot_partial_dependence
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

from scipy.stats import entropy

from data import x_grid_data

def plot_gam(gam, feature_names, target_name, terms_per_row=5):
    plt.figure(figsize=(15, 15), dpi=150)

    k = len(gam.terms) - gam.fit_intercept

    for i in range(k):
        if gam.terms[i].istensor and len(gam.terms[i].feature)==2:
            ax = plt.subplot(k // terms_per_row + 1, terms_per_row, i+1, projection='3d')
            XX = gam.generate_X_grid(term=i, meshgrid=True)
            Z = gam.partial_dependence(term=i, X=XX, meshgrid=True)
            ax.plot_surface(XX[0], XX[1], Z, cmap='viridis')
            ax.set_xlabel(feature_names[gam.terms[i].feature[0]])
            ax.set_ylabel(feature_names[gam.terms[i].feature[1]])
        else:
            ax = plt.subplot(k // terms_per_row + 1, terms_per_row, i+1)
            XX = gam.generate_X_grid(term=i)
            plt.plot(XX[:, gam.terms[i].feature], gam.partial_dependence(term=i, X=XX))
            plt.plot(XX[:, gam.terms[i].feature], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
            plt.xlabel(feature_names[gam.terms[i].feature])
            if i % terms_per_row == 0:
                plt.ylabel('effect on '+ target_name)
            plt.tight_layout()


SCATTER_STYLE_NONE = {
    'color' : 'black', 
    'label' : 'none', 
    'marker' : '*'
}

SCATTER_STYLE_SPHERE = {
    'facecolors' : 'red',
    's' : 20,
    'label' : 'sphere',
    'marker' : '.'
}

SCATTER_STYLE_VESICLE = {
    'facecolor' : 'none', 
    'edgecolors' : 'green',
    'label': 'vesicle', 
    'marker': 'o',
    's' : 60
}

SCATTER_STYLE_WORM = {
    'facecolors' : 'blue', 
    'label' : 'worm',
    'marker' : 'x'
}

SCATTER_STYLE_OTHER = {
    'edgecolors' : 'black', 
    'facecolor' : 'None', 
    'marker' : 's', 
    'label' : 'other'
}

def scatter_phases(sample, x, y, none=True, sphere=True, worm=True, vesicle=True, other=True, ax=None):
    """Plots phases of sample points in a scatter plot such that each phase is a combination of
    markers of the presence of individual morphologies.

    :param phases: pd.DataFrame of shape (n, l) that contains one phase vector per data point, i.e.,
                   four columns corresponding to sphere, worm, vesicle, and other
    :param x: x-coordinates for the scatter plot
    :param y: y-coordinates for the scatter plot
    """
    ax = plt.gca() if ax is None else ax
    if none and sum(sample.sum(axis=1)==0)>1:
        ax.scatter(x[sample.sum(axis=1)==0], y[sample.sum(axis=1)==0], **SCATTER_STYLE_NONE)
    if worm: ax.scatter(x[sample.worm==1], y[sample.worm==1], **SCATTER_STYLE_WORM)
    if vesicle: ax.scatter(x[sample.vesicle==1], y[sample.vesicle==1], **SCATTER_STYLE_VESICLE)
    if sphere: ax.scatter(x[sample.sphere==1], y[sample.sphere==1], **SCATTER_STYLE_SPHERE)
    if other: ax.scatter(x[sample.other==1], y[sample.other==1], **SCATTER_STYLE_OTHER)

def plot_morphology_contour(xx1, xx2, probs, col, ax):
    if probs.max()>=0.5:
        ax.contourf(xx1, xx2, probs.reshape(xx1.shape), levels=[0.5, 1], colors=col, alpha=0.2, zorder=0.5)
        if probs.min()<0.5:
            ax.contour(xx1, xx2, probs.reshape(xx1.shape), levels=[0.5, 1], colors=col, zorder=0.6)

def meshgrid_around_sample(x1, x2, resolution=100, rel_margin = 0.1):
    x1_width = x1.max()-x1.min()
    x2_width = x2.max()-x2.min()
    x1_min = x1.min()-rel_margin*x1_width
    x1_max = x1.max()+rel_margin*x1_width
    x2_min = x2.min()-rel_margin*x2_width
    x2_max = x2.max()+rel_margin*x2_width
    xx1 = np.linspace(x1_min, x1_max, resolution)
    xx2 = np.linspace(x2_min, x2_max, resolution)
    xx1, xx2 = np.meshgrid(xx1, xx2, indexing='xy')
    return xx1, xx2

def x_grid_data_around_sample(sample, x1_var='conc', x2_var='dp_core', resolution=100, rel_margin = 0.1):
    """
    Generates grid of model input points around sample, varying only
    two variables and setting the rest equal to values in first sample point.
    """
    x1 = sample[x1_var]
    x2 = sample[x2_var]
    xx1, xx2 = meshgrid_around_sample(x1, x2, resolution=resolution, rel_margin=rel_margin)
    prototype = sample.iloc[0]
    return xx1, xx2, x_grid_data(prototype, xx1, xx2)

def plot_marginal_morphology_contours(xx1, xx2, yy_hat, ax=None):
    ax = plt.gca() if ax is None else ax
    plot_morphology_contour(xx1, xx2, yy_hat[:, 0], 'red', ax=ax)
    plot_morphology_contour(xx1, xx2, yy_hat[:, 1], 'blue', ax=ax)
    plot_morphology_contour(xx1, xx2, yy_hat[:, 2], 'green', ax=ax)


SCATTER_STYLE_TRAINING = {
    'marker' : 'D',
    'facecolor' : 'none',
    'edgecolor' : 'black',
    's' : 120,
    'label' : 'training'
}

def plot_active_learning_phase_diagrams(exp, k = [0, 1, 4, 7, 10], figsize=None, resolution=100, verbose=True):
    figsize = (len(k)*12/5, 5.5) if figsize is None else figsize
    fig, axs = plt.subplots(2, len(k), figsize=figsize, sharex=True, sharey=True, tight_layout=True)
    xx1, xx2, grid_points = x_grid_data_around_sample(exp.x_test[0], 'conc', 'dp_core', resolution=resolution)

    for j in range(len(k)):
        if verbose:
            print('.', end='')
        yy_hat = exp.fits[k[j]].predict_proba(grid_points)
        plot_marginal_morphology_contours(xx1, xx2, yy_hat, ax=axs[0, j])
        scatter_phases(exp.y_test[0], exp.x_test[0]['conc'], exp.x_test[0]['dp_core'], ax=axs[0, j])

        HH = entropy(yy_hat, axis=1).reshape(xx1.shape)
        entropy_cp = axs[1, j].contourf(xx1, xx2, HH, levels=100, cmap='YlOrBr', vmin=0, vmax=2)
        # plt.colorbar(entropy_cp, ax=axs[1, j])
    if verbose:
        print()

    for i in range(1, len(k)):
        axs[0, i].scatter(exp.x_train[k[i]][-k[i]:]['conc'], exp.x_train[k[i]][-k[i]:]['dp_core'], **SCATTER_STYLE_TRAINING)
        scatter_phases(exp.y_train[k[i]][-k[i]:], exp.x_train[k[i]][-k[i]:]['conc'], exp.x_train[k[i]][-k[i]:]['dp_core'], ax=axs[1, i])

    for j in range(len(k)):
        axs[0, j].set_title(f'$m={k[j]}$ (err ${exp.results_.iloc[k[j]]["full_test_error"]: .2f}$)')
        axs[1, j].set_xlabel('conc')

    axs[0, 0].set_ylabel('dp_core')
    axs[1, 0].set_ylabel('dp_core')

    axs[1, 0].scatter([], [], **SCATTER_STYLE_SPHERE)
    axs[1, 0].scatter([], [], **SCATTER_STYLE_WORM)
    axs[1, 0].scatter([], [], **SCATTER_STYLE_VESICLE)
    axs[1, 0].legend()

    sc_tr = axs[0, 0].scatter([], [], **SCATTER_STYLE_TRAINING)
    axs[0, 0].legend(handles=[sc_tr])
    return fig, axs

def plot_model_individual_partial_dependency(est, X, features, num_importance, ax=None, n_jobs=12):
    ax = plt.gca() if ax is None else ax
    common_params = {
        "n_jobs": n_jobs,
    }
    plot_partial_dependence(est, 
                            features=features[:num_importance], 
                            X=X,
                            ax=ax,
                            kind = 'average',
                            **common_params)

def estimator_feature_importance(est, X):
    feature_importance = np.array(est.feature_importances_)
    sorted_idx = feature_importance.argsort()
    sorted_feature = list(X.columns[sorted_idx])[::-1]
    importance_scores = feature_importance[sorted_idx][::-1]
    return sorted_feature, importance_scores

def plot_model_partial_dependency(est, X, num_importance, nrow=3, figsize=(12, 12), sharey='row', n_jobs=12):
    fig, axs = plt.subplots(ncols=num_importance, nrows=nrow, figsize=figsize, sharey=sharey, tight_layout=True)
    for j in range(3):
        col_indx = -4 + j
        sorted_feature, importance_scores = estimator_feature_importance(est[j], X)
        plot_model_individual_partial_dependency(est = est[j],
                                                X=X.iloc[:, :col_indx],
                                                features=sorted_feature,
                                                num_importance = num_importance,
                                                ax=axs[j, :],
                                                n_jobs = n_jobs)
        for i in range(num_importance):
            text = sorted_feature[i] + ': ' + str(round(importance_scores[i], 4))
            axs[j, i].text(0.5, 0.95, text, horizontalalignment='center', verticalalignment='center', transform=axs[j, i].transAxes)
            axs[j, i].set_xlabel(None)
            axs[j, i].set_ylabel(None)
    axs[0,0].set_ylabel('Sphere')
    axs[1,0].set_ylabel('Worm')
    axs[2,0].set_ylabel('Vesicle')
    fig.supxlabel(r'$X_i$')