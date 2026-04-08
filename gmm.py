from sklearn.mixture import GaussianMixture
import numpy as np

X = np.random.rand(100, 2)

gmm = GaussianMixture(n_components=2)
gmm.fit(X)

print("GMM Means:", gmm.means_)
