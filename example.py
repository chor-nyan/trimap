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
from utils import calculate_AUC
import keras
import umap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cols = cm.tab10(np.linspace(0, 1, 10))

# (X, L), (_, _) = keras.datasets.fashion_mnist.load_data()
#
X = np.load('./data/mnist_X.npy')
L = np.load('./data/mnist_L.npy').flatten()

# mnist = fetch_openml('mnist_784', version=1,)
# X = mnist.data
# L = mnist.target

# X = X[:10000].reshape((10000, 28*28))
# X = X / 255.
# L = L[:10000]

X = X[:10000]
L = L[:10000].astype(int)
print("Dataset size = ({},{})".format(X.shape[0], X.shape[1]))


# y_trimap = trimap.TRIMAP(verbose=True).fit_transform(X)

# # Outlier
# index = 9423
# c = np.random.normal(size=X.shape[1]) # create a random direction
Xo = X.copy()
# Xo[index,:] += 5.0 * c

yo_trimap = trimap.TRIMAP(verbose=True, hub='mp3').fit_transform(Xo)
# yo_trimap = umap.UMAP().fit_transform(X)

plt.scatter(yo_trimap[:, 0], yo_trimap[:, 1], s=0.1, c=cols[L, :])
# plt.scatter(yo_trimap[index,0], yo_trimap[index,1], s=80, c='red', marker='x')
plt.show()


# yo_pca = PCA(n_components = 2).fit_transform(Xo)
# plt.scatter(yo_pca[:,0], yo_pca[:,1], s=0.1, c=cols[L,:])
# plt.scatter(yo_pca[index,0], yo_pca[index,1], s=80, c='red', marker='x')
# plt.show()

# AUC
auc = calculate_AUC(Xo, yo_trimap)
print("AUC: ", auc)

# 1-NN
X_train, X_test, Y_train, Y_test = train_test_split(yo_trimap, L, random_state=0)
knc = KNeighborsClassifier(n_neighbors=1)
knc.fit(X_train, Y_train)

Y_pred = knc.predict(X_test)
score = knc.score(X_test, Y_test)
print("1-NN: ", score)


scale = False
if scale:
    flag1 = yo_trimap[:, 0] > -150
    flag2 = yo_trimap[:, 1] < 130
    flag = flag1 & flag2
    plt.scatter(yo_trimap[flag,0], yo_trimap[flag,1], s=0.1, c=cols[L[flag],:])
    plt.show()
# flag = [True] * Xo.shape[0]
# flag[index] = False
# plt.scatter(yo_trimap[flag,0], yo_trimap[flag,1], s=0.1, c=cols[L[flag],:])


# Normal: 0.12329153085829356
# MP: 0.14389730201672235

# Normal: 0.13352001439658828
# MP2: 0.14220084078308895
# MP3: 0.1495966750861319
# SNN1: 0.18295933669763098