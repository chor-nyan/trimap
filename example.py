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

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data[:70000, :]
L = mnist.target[:70000].astype(int)
print("Dataset size = ({},{})".format(X.shape[0], X.shape[1]))

# y_pca = PCA(n_components = 2).fit_transform(X)
# y_tsne = TSNE().fit_transform(X)
# y_umap = umap.UMAP().fit_transform(X)
y_trimap = trimap.TRIMAP(verbose=False).fit_transform(X)

# plot_results([y_tsne,y_umap,y_trimap,y_pca], cols[L,:], extension='(orig)')
plt.scatter(y_trimap[:,0], y_trimap[:,1], s=0.1, c=cols[L,:])
plt.show()