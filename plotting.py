from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

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
    if sphere: ax.scatter(x[sample.sphere==1], y[sample.sphere==1], **SCATTER_STYLE_SPHERE)
    if worm: ax.scatter(x[sample.worm==1], y[sample.worm==1], **SCATTER_STYLE_WORM)
    if vesicle: ax.scatter(x[sample.vesicle==1], y[sample.vesicle==1], **SCATTER_STYLE_VESICLE)
    if other: ax.scatter(x[sample.other==1], y[sample.other==1], **SCATTER_STYLE_OTHER)
