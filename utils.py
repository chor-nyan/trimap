# https://github.com/eamid/examples/blob/master/utils.py

import matplotlib.pyplot as plt
from sklearn import metrics
from hub_toolbox.distances import euclidean_distance
import numpy as np
import numba
# import hub_toolbox_python3.hub_toolbox.centering

def plot_results(maps, labels, index=[], extension=''):
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(15)
    ax1 = fig.add_subplot(2, 3, 1); ax1.axis('off')
    ax2 = fig.add_subplot(2, 3, 2); ax2.axis('off')
    ax3 = fig.add_subplot(2, 3, 3); ax3.axis('off')
    ax4 = fig.add_subplot(2, 3, 4); ax4.axis('off')
    ax5 = fig.add_subplot(2, 3, 5); ax5.axis('off')
    ax1.scatter(maps[0][:,0], maps[0][:,1], s=0.1, c=labels)
    ax2.scatter(maps[1][:,0], maps[1][:,1], s=0.1, c=labels)
    ax3.scatter(maps[2][:,0], maps[2][:,1], s=0.1, c=labels)
    ax4.scatter(maps[3][:,0], maps[3][:,1], s=0.1, c=labels)
    ax5.scatter(maps[4][:,0], maps[4][:,1], s=0.1, c=labels)
    if index:
        ax1.scatter(maps[0][index,0], maps[0][index,1], s=80, c='red', marker='x')
        ax2.scatter(maps[1][index,0], maps[1][index,1], s=80, c='red', marker='x')
        ax3.scatter(maps[2][index,0], maps[2][index,1], s=80, c='red', marker='x')
        ax4.scatter(maps[3][index,0], maps[3][index,1], s=80, c='red', marker='x')
        ax5.scatter(maps[4][index,0], maps[4][index,1], s=80, c='red', marker='x')
    ax1.title.set_text('t-SNE ' + extension)
    ax2.title.set_text('UMAP ' + extension)
    ax3.title.set_text('TriMap ' + extension)
    ax4.title.set_text('hub_Trimap ' + extension)
    ax5.title.set_text('PCA ' + extension)
    plt.show()

@numba.jit()
def sort_D(D, k=100):
    n = D.shape[0]  # サンプルサイズ
    sortD = np.zeros((n, k),)
    sortD_idx = np.zeros((n, k), dtype=int)

    for i in np.arange(0, n):
        d_vec = D[i, :]  # i-th row
        v = np.argsort(d_vec)  # 昇順にソートした配列のインデックス
        sortD_idx[i, :] = v[1:k + 1]  # 距離が短い順にk個選ぶ（自分を除く）
        sortD[i, :] = d_vec[sortD_idx[i, :]]

    return sortD, sortD_idx

@numba.jit()
def calculate_AUC(X, embed):
    # FPR, TPRを近傍数kごとに算出
    # 高次元の近傍数は20, 低次元の近傍数は1~100
    # fpr.shape = (100, )
    # D_XとD_embedは昇順にソートされた距離行列（0は除く）

    k_high = 20
    Ks = 100  # 低次元での近傍数の数
    n = X.shape[0]

    r_i = 20
    k_i = [a for a in range(1, 101)]

    D_X = euclidean_distance(X)
    D_embed = euclidean_distance(embed)

    sortD_X, sortD_X_idx = sort_D(D_X, k=20)
    sortD_embed, sortD_embed_idx = sort_D(D_embed, k=100)  # n×100-matrix
    n_precision = np.zeros((n, len(k_i)), dtype=float)  # precision
    n_recall = np.zeros((n, len(k_i)), dtype=float)  # recall
    print(n_precision.shape, n_recall.shape)
    for i in range(n):
        for j in range(100):
            tp = np.intersect1d(sortD_X_idx[i, :], sortD_embed_idx[i, :j + 1])
            # fp = np.setdiff1d(sortD_X_idx[i, :], sortD_embed_idx[i, :j + 1])
            if len(tp) > 0:
                n_precision[i, j] += len(tp) / (j + 1.)
                n_recall[i, j] += len(tp) / 20.
            # if len(fp) > 0:
            #     n_recall[i, j] += len(fp) / 20.

    # print(n_tp, n_fp)


    average_precision = np.mean(n_precision, axis=0)
    average_recall = np.mean(n_recall, axis=0)

    # print(fpr, tpr)

    auc = metrics.auc(average_recall, average_precision)

    return auc

