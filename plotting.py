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
