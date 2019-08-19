# https://github.com/eamid/examples

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from utils import plot_results
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# import umap
import trimap

cols = cm.tab10(np.linspace(0, 1, 10))

X = np.load('./data/mnist_X.npy')
L = np.load('./data/mnist_L.npy').flatten()
print("Dataset size = ({},{})".format(X.shape[0],X.shape[1]))


# y_trimap = trimap.TRIMAP(verbose=True).fit_transform(X)

# Outlier
index = 9423
c = np.random.normal(size=X.shape[1]) # create a random direction
Xo = X.copy()
Xo[index,:] += 5.0 * c

yo_trimap = trimap.TRIMAP(verbose=True, hub='ls').fit_transform(Xo)




plt.scatter(yo_trimap[:,0], yo_trimap[:,1], s=0.1, c=cols[L,:])
plt.scatter(yo_trimap[index,0], yo_trimap[index,1], s=80, c='red', marker='x')
plt.show()

# flag = [True] * Xo.shape[0]
# flag[index] = False
# plt.scatter(yo_trimap[flag,0], yo_trimap[flag,1], s=0.1, c=cols[L[flag],:])
