#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

TriMap: Dimensionality Reduction Using Triplet Constraints

@author: Ehsan Amid <eamid@ucsc.edu>

"""


from sklearn.base import BaseEstimator
import numba
from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors as knn
from sklearn.decomposition import TruncatedSVD
import numpy as np
import time
import datetime
import sys
from sklearn.decomposition import PCA
from skhubness.neighbors import kneighbors_graph
import hub_toolbox
from hub_toolbox.distances import euclidean_distance
from skhubness.reduction import shared_neighbors

if sys.version_info < (3,):
    range = xrange

bold = "\033[1m"
reset = "\033[0;0m"

@numba.njit()
def euclid_dist(x1, x2):
    """
    Fast Euclidean distance calculation between two vectors.

    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += (x1[i]-x2[i])**2
    return np.sqrt(result)

@numba.jit()
def make_mutual(neighbour_matrix):
    for i in range(neighbour_matrix.shape[0]):
        for j in range(neighbour_matrix.shape[1]):
            if i not in neighbour_matrix[neighbour_matrix[i, j]]:
                neighbour_matrix[i, j] = neighbour_matrix.shape[0] + 1
    return neighbour_matrix

# Define KNN graph information function
@numba.jit()
def KNN_Info(D, k, geodesic=None):
    (n, _) = D.shape
    sortD = np.zeros((n, k), )
    sortD_idx = np.zeros((n, k), dtype=int)

    for i in np.arange(0, n):
        d_vec = D[i, :]  # i-th row
        v = np.argsort(d_vec)  # 昇順にソートした配列のインデックス
        sortD_idx[i, :] = v[1:k + 1]  # 距離が短い順にk個選ぶ（自分を除く）
        sortD[i, :] = d_vec[sortD_idx[i, :]]

    deg = np.zeros((1, n), dtype=int)
    v = sortD_idx.reshape(1, n * k)
    for i in np.arange(0, n):
        (_, v0) = np.where(v == i)  # vは二次元配列で, タプルの1つめの要素はすべて0になる
        deg[:, i] = len(v0)

    return sortD, sortD_idx

@numba.jit()
def calculate_deg(nbrs):
    n = nbrs.shape[0]
    deg = np.array([0] * n)
    v = nbrs.reshape(1, n * nbrs.shape[1])
    for i in range(n):
        (_, v0) = np.where(v == i)
        deg[i] = len(v0)

    mean_deg = np.mean(deg)
    var_deg = np.var(deg)

    return deg, mean_deg, var_deg



@numba.njit()
def rejection_sample(n_samples, max_int, rejects):
    """
    Samples "n_samples" integers from a given interval [0,max_int] while
    rejecting the values that are in the "rejects".

    """
    result = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        reject_sample = True
        while reject_sample:
            j = np.random.randint(max_int)
            for k in range(i):
                if j == result[k]:
                    break
            for k in range(rejects.shape[0]):
            	if j == rejects[k]:
            		break
            else:
                reject_sample = False
        result[i] = j
    return result


@numba.njit('i8[:,:](f8[:,:],i8[:,:], i8,i8)', parallel=True, nogil=True)
def sample_knn_triplets(P, nbrs, n_inlier, n_outlier):
    """
    Sample nearest neighbors triplets based on the similarity values given in P

    Input
    ------

    nbrs: Nearest neighbors indices for each point. The similarity values 
        are given in matrix P. Row i corresponds to the i-th point.

    P: Matrix of pairwise similarities between each point and its neighbors 
        given in matrix nbrs

    n_inlier: Number of inlier points

    n_outlier: Number of outlier points

    Output
    ------

    triplets: Sampled triplets
    """
    n, n_neighbors = nbrs.shape
    triplets = np.empty((n * n_inlier * n_outlier, 3), dtype=np.int64)
    for i in range(n):
        sort_indices = np.argsort(-P[i,:])
        for j in range(n_inlier):
            sim = nbrs[i,sort_indices[j+1]]
            samples = rejection_sample(n_outlier, n, sort_indices[:j+2])
            for k in range(n_outlier):
                index = i * n_inlier * n_outlier + j * n_outlier + k
                out = samples[k]
                triplets[index,0] = i
                triplets[index,1] = sim
                triplets[index,2] = out
    return triplets



# @numba.njit('f8[:,:](f8[:,:],i8,f8[:])', parallel=True, nogil=True)
@numba.jit()
def sample_random_triplets(X, n_random, sig=None, P=None):
    """
    Sample uniformly random triplets

    Input
    ------

    X: Instance matrix

    n_random: Number of random triplets per point

    sig: Scaling factor for the distances

    Output
    ------

    rand_triplets: Sampled triplets
    """
    n = X.shape[0]
    rand_triplets = np.empty((n * n_random, 4), dtype=np.float64)
    for i in range(n):
        for j in range(n_random):
            sim = np.random.choice(n)
            while sim == i:
                sim = np.random.choice(n)
            out = np.random.choice(n)
            while out == i or out == sim:
                out = np.random.choice(n)

            if not sig is None:
                p_sim = np.exp(-euclid_dist(X[i,:],X[sim,:])**2/(sig[i] * sig[sim]))
            elif not P is None:
                p_sim = P[i, sim]

            if p_sim < 1e-20:
                p_sim = 1e-20

            if not sig is None:
                p_out = np.exp(-euclid_dist(X[i,:],X[out,:])**2/(sig[i] * sig[out]))
            elif not P is None:
                p_out = P[i, out]

            if p_out < 1e-20:
                p_out = 1e-20
            if p_sim < p_out:
                sim, out = out, sim
                p_sim, p_out = p_out, p_sim
            rand_triplets[i * n_random + j,0] = i
            rand_triplets[i * n_random + j,1] = sim
            rand_triplets[i * n_random + j,2] = out
            rand_triplets[i * n_random + j,3] = p_sim/p_out
    return rand_triplets


@numba.njit('f8[:,:](f8[:,:],f8[:],i8[:,:])', parallel=True, nogil=True)
def find_p(distances, sig, nbrs):
    """
    Calculates the similarity matrix P

    Input
    ------

    distances: Matrix of pairwise distances

    sig: Scaling factor for the distances

    nbrs: Nearest neighbors

    Output
    ------

    P: Pairwise similarity matrix
    """
    n, n_neighbors = distances.shape
    P = np.zeros((n,n_neighbors), dtype=np.float64)
    for i in range(n):
        for j in range(n_neighbors):
            P[i,j] = np.exp(-distances[i,j]**2/sig[i]/sig[nbrs[i,j]])
    return P


@numba.njit('f8[:](i8[:,:],f8[:,:],i8[:,:],f8[:],f8[:])',parallel=True, nogil=True)
def find_weights(triplets, P, nbrs, distances, sig):
    """
    Calculates the weights for the sampled nearest neighbors triplets

    Input
    ------

    triplets: Sampled triplets

    P: Pairwise similarity matrix

    nbrs: Nearest neighbors

    distances: Matrix of pairwise distances

    sig: Scaling factor for the distances

    Output
    ------

    weights: Weights for the triplets
    """
    n_triplets = triplets.shape[0]
    weights = np.empty(n_triplets, dtype=np.float64)
    for t in range(n_triplets):
        i = triplets[t, 0]
        sim = 0
        while(nbrs[i, sim] != triplets[t,1]):
            sim += 1
        p_sim = P[i,sim]
        p_out = np.exp(-distances[t]**2/(sig[i] * sig[triplets[t,2]]))
        if p_out < 1e-20:
            p_out = 1e-20
        weights[t] = p_sim/p_out
    return weights

def generate_triplets(X, n_inlier, n_outlier, n_random, fast_trimap = True, weight_adj = True, verbose = True, hub = 'mp'):
    n, dim = X.shape
    if dim > 100:
        X = TruncatedSVD(n_components=100, random_state=0).fit_transform(X)
        dim = 100
    exact = n <= 10000
    n_extra = min(max(n_inlier, 150),n)

    if hub == 'mp1':  # hubness reductionをtriplet選択のみに使用
        neigbour_graph = kneighbors_graph(X, n_neighbors=n_extra, mode='distance', hubness='mutual_proximity',
                                          hubness_params={'method': 'normal'})
        nbrs = neigbour_graph.indices.astype(int).reshape((X.shape[0], n_extra))
        # distances = neigbour_graph.data.reshape((X.shape[0], n_extra))

        flag = nbrs.tolist()

        D = euclidean_distance(X)
        D = np.array([D[i][flag[i]] for i in range(D.shape[0])])

        distances = D

        if verbose:
            print("hubness reduction with {}".format(hub))

    elif hub == 'mp2':  # 類似度Pを１−Dmpにする

        D = euclidean_distance(X)
        D_mp = hub_toolbox.global_scaling.mutual_proximity_gaussi(D=D, metric='distance')

        # make knn graph
        distances, nbrs = KNN_Info(D_mp, n_extra)

        if verbose:
            print("hubness reduction with {}".format(hub))

    elif hub == 'mp3_gauss':  # secondary distanceで類似度を計算
        D = euclidean_distance(X)
        D_mp = hub_toolbox.global_scaling.mutual_proximity_gaussi(D=D, metric='distance')

        # make knn graph
        distances, nbrs = KNN_Info(D_mp, n_extra)

        if verbose:
            print("hubness reduction with {}".format(hub))

    elif hub == 'mp3_emp':  # secondary distanceで類似度を計算
        D = euclidean_distance(X)
        D_mp = hub_toolbox.global_scaling._mutual_proximity_empiric_full(D=D, metric='distance')

        # make knn graph
        distances, nbrs = KNN_Info(D_mp, n_extra)

        if verbose:
            print("hubness reduction with {}".format(hub))

    elif hub == 'mp4':  # 謎
        neigbour_graph = kneighbors_graph(X, n_neighbors=n_extra, mode='distance', hubness='mutual_proximity',
                                          hubness_params={'method': 'normal'})
        nbrs = neigbour_graph.indices.astype(int).reshape((X.shape[0], n_extra))
        distances = neigbour_graph.data.reshape((X.shape[0], n_extra))

        if verbose:
            print("hubness reduction with {}".format(hub))



    elif hub == 'ls1':
        neigbour_graph = kneighbors_graph(X, n_neighbors=n_extra, mode='distance', hubness='local_scaling')
        nbrs = neigbour_graph.indices.astype(int).reshape((X.shape[0], n_extra))
        # distances = neigbour_graph.data.reshape((X.shape[0], n_extra))

        flag = nbrs.tolist()

        D = euclidean_distance(X)
        D = np.array([D[i][flag[i]] for i in range(D.shape[0])])

        distances = D

        if verbose:
            print("hubness reduction with {}".format(hub))

    elif hub == 'ls2':
        D = euclidean_distance(X)
        D_ls = hub_toolbox.local_scaling.local_scaling(D=D, k=10, metric='distance')

        # make knn graph
        distances, nbrs = KNN_Info(D_ls, n_extra)

        if verbose:
            print("hubness reduction with {}".format(hub))

    elif hub == 'dsl':
        neigbour_graph = kneighbors_graph(X, n_neighbors=n_extra, mode='connectivity', hubness='dsl')
        nbrs = neigbour_graph.indices.astype(int).reshape((X.shape[0], n_extra))
        # flag = neigbour_graph.data.reshape((X.shape[0], n_extra))
        flag = nbrs.tolist()

        D = euclidean_distance(X)
        D = np.array([D[i][flag[i]] for i in range(D.shape[0])])

        distances = D

        # D = np.empty((X.shape[0], n_extra, dtype=np.float64)
        # for i in range(X.shape[0]):
        #     for j in range(n_extra):
        #         D[i, j] = euclid_dist(X[i, :], X[nbrs[i][j]])
        #         np.sqrt(np.sum((X[triplets[t, 0], :] - X[triplets[t, 2], :]) ** 2))

        if verbose:
            print("hubness reduction with {}".format(hub))

    elif hub == 'mutual':
        # D = euclidean_distance(X)
        # # make knn graph
        # _, nbrs = KNN_Info(D_mp, n_extra)

        knn_tree = knn(n_neighbors=n_extra, algorithm='auto').fit(X)
        distances, nbrs = knn_tree.kneighbors(X)

        nbrs = make_mutual(nbrs)
        # a = nbrs == X.shape[0] + 1
        # print(a)

    elif hub == 'SNN1' or hub == 'SNN2':
        D = euclidean_distance(X)
        D_snn = hub_toolbox.shared_neighbors.shared_nearest_neighbors(D=D, metric='distance')

        # snn = shared_neighbors(k=10, metric='euclidean')
        # D_snn = snn.fit_tr(X)

        # make knn graph
        distances, nbrs = KNN_Info(D_snn, n_extra)

        if verbose:
            print("hubness reduction with {}".format(hub))


    elif exact: # do exact knn search
        knn_tree = knn(n_neighbors=n_extra, algorithm='auto').fit(X)
        distances, nbrs = knn_tree.kneighbors(X)

        # print(nbrs)
    elif fast_trimap: # use annoy
        tree = AnnoyIndex(dim, metric='euclidean')
        for i in range(n):
            tree.add_item(i, X[i,:])
        tree.build(10)
        nbrs = np.empty((n,n_extra), dtype=np.int64)
        distances = np.empty((n,n_extra), dtype=np.float64)
        dij = np.empty(n_extra, dtype=np.float64)
        for i in range(n):
            nbrs[i,:] = tree.get_nns_by_item(i, n_extra)
            for j in range(n_extra):
                dij[j] = euclid_dist(X[i,:], X[nbrs[i,j],:])
            sort_indices = np.argsort(dij)
            nbrs[i,:] = nbrs[i,sort_indices]
            # for j in range(n_extra):
            #     distances[i,j] = tree.get_distance(i, nbrs[i,j])
            distances[i,:] = dij[sort_indices]
    else:
        n_bf = 10
        n_extra += n_bf
        knn_tree = knn(n_neighbors= n_bf, algorithm='auto').fit(X)
        _, nbrs_bf = knn_tree.kneighbors(X)
        nbrs = np.empty((n,n_extra), dtype=np.int64)
        nbrs[:,:n_bf] = nbrs_bf
        tree = AnnoyIndex(dim, metric='euclidean')
        for i in range(n):
            tree.add_item(i, X[i,:])
        tree.build(100)
        distances = np.empty((n,n_extra), dtype=np.float64)
        dij = np.empty(n_extra, dtype=np.float64)
        for i in range(n):
            nbrs[i,n_bf:] = tree.get_nns_by_item(i, n_extra-n_bf)
            unique_nn = np.unique(nbrs[i,:])
            n_unique = len(unique_nn)
            nbrs[i,:n_unique] = unique_nn
            for j in range(n_unique):
                dij[j] = euclid_dist(X[i,:], X[nbrs[i,j],:])
            sort_indices = np.argsort(dij[:n_unique])
            nbrs[i,:n_unique] = nbrs[i,sort_indices]
            distances[i,:n_unique] = dij[sort_indices]
    if verbose:
        print("found nearest neighbors")
    # if hub == 'ls':
    # #     sig = np.array([1.]*X.shape[0])
    # else:
    if hub == 'mp2':
        P = 1 - distances  # (n, k)

    # elif hub == 'mp3':
    #     sig = np.median(D_mp[np.triu_indices(D_mp.shape[0], k=1)])
    #     sig = np.array([sig] * D_mp.shape[0])
    #     P = find_p(distances, sig, nbrs)

    else:
        sig = np.maximum(np.mean(distances[:, 10:20], axis=1), 1e-20) # scale parameter
        P = find_p(distances, sig, nbrs)
    # if hub == 'ls':
    #     P = -np.log(P)
    #     P = np.sqrt(P)
    #     P = 1 - P
    triplets = sample_knn_triplets(P, nbrs, n_inlier, n_outlier)
    print("tri_shape", triplets[0], triplets[0][2])
    n_triplets = triplets.shape[0]
    # if hub == 'mp':
    #     outlier_dist

    # if not hub == 'mp':
    #
    outlier_dist = np.empty(n_triplets, dtype=np.float64)
    # if hub == 'mp':
    #     for t in range(n_triplets):
    #         outlier_dist[t] = D_mp[triplets[t][0], triplets[t][2]]
    # el

    if hub == 'mp2' or hub == 'SNN1' or hub == 'ls2':
        pass

    elif 'mp3' in hub:
        for t in range(n_triplets):
            outlier_dist[t] = D_mp[triplets[t][0], triplets[t][2]]

    elif hub == 'SNN2':
        for t in range(n_triplets):
            outlier_dist[t] = D_snn[triplets[t][0], triplets[t][2]]

    elif exact or  not fast_trimap:
        for t in range(n_triplets):
            outlier_dist[t] = np.sqrt(np.sum((X[triplets[t,0],:] - X[triplets[t,2],:])**2))
    else:
        for t in range(n_triplets):
            outlier_dist[t] = euclid_dist(X[triplets[t,0],:], X[triplets[t,2],:])
            # outlier_dist[t] = tree.get_distance(triplets[t,0], triplets[t,2])

    if hub == 'mp2' or hub == 'SNN1' or hub == 'ls2':
        if hub == 'SNN1':
            D_mp = D_snn
        elif hub == 'ls2':
            D_mp = D_ls

        n_triplets = triplets.shape[0]
        weights = np.empty(n_triplets, dtype=np.float64)
        print("P and triplets' shape", triplets)
        P = 1 - D_mp  # (n, n)
        for t in range(n_triplets):
            i = triplets[t, 0]
            p_sim = P[i, triplets[t, 1]]
            p_out = P[i, triplets[t, 2]]
            if p_out < 1e-20:
                p_out = 1e-20
            weights[t] = p_sim / p_out
    else:
        weights = find_weights(triplets, P, nbrs, outlier_dist, sig)


    if hub == 'weight':
        deg, mean_deg, var_deg = calculate_deg(nbrs)
        var_deg = max(var_deg, 1e-20)
        # hubness_score = (deg - mean_deg) / var_deg
        # hs_med = np.mean(hubness_score)
        hs_med = np.median(deg)
        hub_weights = np.exp(- deg/hs_med)
        # hub_weights = np.exp(- hubness_score)

        # print(hubness_score)

        m = hub_weights.shape[0]
        l = n_inlier * n_outlier

        for i in range(m):
            for j in range(l):
                weights[i * l:i * l + j] = hub_weights[i] * weights[i * l:i * l + j]


    print('out_dist: ', outlier_dist)

    if n_random > 0:
        if hub == 'mp2' or hub == 'SNN1' or hub == 'ls2':
            rand_triplets = sample_random_triplets(X, n_random, P=P)  # P: (n, n)

        else:
            rand_triplets = sample_random_triplets(X, n_random, sig=sig)

        rand_weights = rand_triplets[:,-1]
        rand_triplets = rand_triplets[:,:-1].astype(np.int64)
        triplets = np.vstack((triplets, rand_triplets))
        weights = np.hstack((weights, rand_weights))
    weights /= np.max(weights)
    weights += 0.0001
    if weight_adj:
        if not isinstance(weight_adj, (int, float)):
            weight_adj = 400.0
        weights = np.log(1 + weight_adj * weights)
        weights /= np.max(weights)
    return (triplets, weights)


@numba.njit('void(f8[:,:],f8[:,:],f8[:,:],f8,i8,i8)', parallel=True, nogil=True)
def update_embedding(Y, grad, vel, lr, iter_num, opt_method):
    n, dim = Y.shape
    if opt_method == 0: # sd
        for i in range(n):
            for d in range(dim):
                Y[i,d] -= lr * grad[i,d]
    elif opt_method == 1: # momentum
        if iter_num > 250:
            gamma = 0.5
        else:
            gamma = 0.3
        for i in range(n):
            for d in range(dim):
                vel[i,d] = gamma * vel[i,d] - lr * grad[i,d] # - 1e-5 * Y[i,d]
                Y[i,d] += vel[i,d]

@numba.njit('void(f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8,i8)', parallel=True, nogil=True)
def update_embedding_dbd(Y, grad, vel, gain, lr, iter_num):
    n, dim = Y.shape
    if  iter_num > 250:
        gamma = 0.8 # moment parameter
    else:
        gamma = 0.5
    min_gain = 0.01
    for i in range(n):
        for d in range(dim):
            gain[i,d] = (gain[i,d]+0.2) if (np.sign(vel[i,d]) != np.sign(grad[i,d])) else np.maximum(gain[i,d]*0.8, min_gain)
            vel[i,d] = gamma * vel[i,d] - lr * gain[i,d] * grad[i,d]
            Y[i,d] += vel[i,d]

                
@numba.njit('f8[:,:](f8[:,:],i8,i8,i8[:,:],f8[:])', parallel=True, nogil=True)
def trimap_grad(Y, n_inlier, n_outlier, triplets, weights):
    n, dim = Y.shape
    n_triplets = triplets.shape[0]
    grad = np.zeros((n, dim), dtype=np.float64)
    y_ij = np.empty(dim, dtype=np.float64)
    y_ik = np.empty(dim, dtype=np.float64)
    n_viol = 0.0
    loss = 0.0
    n_knn_triplets = n * n_inlier * n_outlier
    for t in range(n_triplets):
        i = triplets[t,0]
        j = triplets[t,1]
        k = triplets[t,2]
        if (t % n_outlier) == 0 or (t >= n_knn_triplets): # update y_ij, d_ij
            d_ij = 1.0
            d_ik = 1.0
            for d in range(dim):
                y_ij[d] = Y[i,d] - Y[j,d]
                y_ik[d] = Y[i,d] - Y[k,d]
                d_ij += y_ij[d]**2
                d_ik += y_ik[d]**2
        else:
            d_ik = 1.0
            for d in range(dim):
                y_ik[d] = Y[i,d] - Y[k,d]
                d_ik += y_ik[d]**2
        if (d_ij > d_ik):
            n_viol += 1.0
        loss += weights[t] * 1.0/(1.0 + d_ik/d_ij)
        w = weights[t]/(d_ij + d_ik)**2
        for d in range(dim):
            gs = y_ij[d] * d_ik * w
            go = y_ik[d] * d_ij * w
            grad[i,d] += gs - go
            grad[j,d] -= gs
            grad[k,d] += go
    last = np.zeros((1,dim), dtype=np.float64)
    last[0] = loss
    last[1] = n_viol
    return np.vstack((grad, last))
    
    
def trimap(X, triplets, weights, n_dims, n_inliers, n_outliers, n_random, lr, n_iters, Yinit,
 weight_adj, fast_trimap, opt_method, verbose, return_seq, hub):
    if verbose:
        t = time.time()
    n, dim = X.shape
    if verbose:
        print("running TriMap on %d points with dimension %d" % (n, dim))
    if triplets[0] is None:
        if verbose:
            print("pre-processing")
        X -= np.min(X)
        X /= np.max(X)
        X -= np.mean(X,axis=0)
        triplets, weights = generate_triplets(X, n_inliers, n_outliers, n_random, fast_trimap, weight_adj, verbose, hub)
        if verbose:
            print("sampled triplets")
    else:
        if verbose:
            print("using stored triplets")
        
    if Yinit is None or Yinit is 'pca':
        Y = 0.01 * PCA(n_components = n_dims).fit_transform(X)
    elif Yinit is 'random':
        Y = np.random.normal(size=[n, n_dims]) * 0.0001
    else:
        Y = Yinit
    if return_seq:
        Y_all = np.zeros((n, n_dims, int(n_iters/10 + 1)))
        Y_all[:,:,0] = Yinit
    C = np.inf
    tol = 1e-7
    n_triplets = float(triplets.shape[0])
    lr = lr * n / n_triplets
    opt_method_index = {'sd':0, 'momentum':1, 'dbd':2}
    if verbose:
        print("running TriMap with " + opt_method)
    vel = np.zeros_like(Y, dtype=np.float64)
    if opt_method_index[opt_method] == 2:
        gain = np.ones_like(Y, dtype=np.float64)

    for itr in range(n_iters):
        old_C = C
        if opt_method_index[opt_method] == 0:
            grad = trimap_grad(Y, n_inliers, n_outliers, triplets, weights)
        else:
            if itr > 250:
                gamma = 0.5
            else:
                gamma = 0.3
            grad = trimap_grad(Y + gamma * vel, n_inliers, n_outliers, triplets, weights)
        C = grad[-1,0]
        n_viol = grad[-1,1]
            
        # update Y
        if opt_method_index[opt_method] < 2:
            update_embedding(Y, grad, vel, lr, itr, opt_method_index[opt_method])
        else:
            update_embedding_dbd(Y, grad, vel, gain, lr, itr)
        
        # update the learning rate
        if opt_method_index[opt_method] < 2:
            if old_C > C + tol:
                lr = lr * 1.01
            else:
                lr = lr * 0.9
        if return_seq and (itr+1) % 10 == 0:
            Y_all[:,:,int((itr+1)/10)] = Y
        if verbose:
            if (itr+1) % 100 == 0:
                print('Iteration: %4d, Loss: %3.3f, Violated triplets: %0.4f' % (itr+1, C, n_viol/n_triplets*100.0))
    if verbose:
        elapsed = str(datetime.timedelta(seconds= time.time() - t))
        print("Elapsed time: %s" % (elapsed))
    if return_seq:
        return (Y_all, triplets, weights)
    else:
        return (Y, triplets, weights)

class TRIMAP(BaseEstimator):
    """
    Dimensionality Reduction Using Triplet Constraints

    Find a low-dimensional repersentation of the data by satisfying the sampled
    triplet constraints from the high-dimensional features.

    Input
    ------

    n_dims: Number of dimensions of the embedding (default = 2)

    n_inliers: Number of inlier points for triplet constraints (default = 10)

    n_outliers: Number of outlier points for triplet constraints (default = 5)

    n_random: Number of random triplet constraints per point (default = 5)

    lr: Learning rate (default = 1000.0)

    n_iters: Number of iterations (default = 400)

    fast_trimap: Use fast nearest neighbor calculation (default = True)

    opt_method: Optimization method ('sd': steepest descent,  'momentum': GD with momentum, 'dbd': GD with momentum delta-bar-delta (default))

    verbose: Print the progress report (default = True)

    weight_adj: Adjusting the weights using a non-linear transformation (default = True)

    return_seq: Return the sequence of maps recorded every 10 iterations (default = False)
    """

    def __init__(self,
                 n_dims=2,
                 n_inliers=10,
                 n_outliers=5,
                 n_random=5,
                 lr=1000.0,
                 n_iters=400,
                 triplets=None,
                 weights=None,
                 verbose=True,
                 weight_adj=True,
                 fast_trimap=True,
                 opt_method='dbd',
                 return_seq=False,
                 hub='mp'
                 ):
        self.n_dims = n_dims
        self.n_inliers = n_inliers
        self.n_outliers = n_outliers
        self.n_random = n_random
        self.lr = lr
        self.n_iters = n_iters
        self.triplets = triplets,
        self.weights = weights
        self.weight_adj = weight_adj
        self.fast_trimap = fast_trimap
        self.opt_method = opt_method
        self.verbose = verbose
        self.return_seq = return_seq
        self.hub = hub

        if self.n_dims < 2:
            raise ValueError('The number of output dimensions must be at least 2.')
        if self.n_inliers < 1:
            raise ValueError('The number of inliers must be a positive number.')
        if self.n_outliers < 1:
            raise ValueError('The number of outliers must be a positive number.')
        if self.n_random < 0:
            raise ValueError('The number of random triplets must be a non-negative number.')
        if self.lr <= 0:
            raise ValueError('The learning rate must be a positive value.')

        if self.verbose:
            print("TRIMAP(n_inliers={}, n_outliers={}, n_random={}, "
                  "lr={}, n_iters={}, weight_adj={}, fast_trimap = {}, opt_method = {}, verbose={}, return_seq={}, hub={})".format(
                  n_inliers, n_outliers, n_random, lr, n_iters, weight_adj, fast_trimap, opt_method, verbose, return_seq, hub))
            if not self.fast_trimap:
                print(bold + "running exact nearest neighbors search. TriMap may be slow!" + reset)

    def fit(self, X, init = None):
        """
        Runs the TriMap algorithm on the input data X

        Input
        ------

        X: Instance matrix

        init: Initial solution
        """
        X = X.astype(np.float64)
        
        self.embedding_, self.triplets, self.weights = trimap(X, self.triplets,
            self.weights, self.n_dims, self.n_inliers, self.n_outliers, self.n_random,
            self.lr, self.n_iters, init, self.weight_adj, self.fast_trimap, self.opt_method, self.verbose, self.return_seq, self.hub)
        return self

    def fit_transform(self, X, init = None):
        """
        Runs the TriMap algorithm on the input data X and returns the embedding

        Input
        ------

        X: Instance matrix

        init: Initial solution
        """
        self.fit(X, init)
        return self.embedding_
    
    def sample_triplets(self, X):
        """
        Samples and stores triplets

        Input
        ------

        X: Instance matrix
        """
        if self.verbose:
            print("pre-processing")
        X = X.astype(np.float64)
        X -= np.min(X)
        X /= np.max(X)
        X -= np.mean(X,axis=0)
        self.triplets, self.weights = generate_triplets(X, self.n_inliers, self.n_outliers, self.n_random, self.fast_trimap, self.weight_adj, self.verbose, self.hub)
        if self.verbose:
            print("sampled triplets")
        
        return self
    
    def del_triplets(self):
        """
        Deletes the stored triplets
        """
        self.triplets = None
        self.weights = None
        
        return self

    def global_score(self, X, Y):
        """
        Global score

        Input
        ------

        X: Instance matrix
        Y: Embedding
        """
        def global_loss_(X, Y):
            X = X - np.mean(X, axis=0)
            Y = Y - np.mean(Y, axis=0)
            A = np.dot(np.dot(X.T, Y), np.linalg.inv(np.dot(Y.T, Y)))
            return np.mean(np.power(X.T - np.dot(A, Y.T), 2))
        n_dims = Y.shape[1]
        Y_pca = PCA(n_components = n_dims).fit_transform(X)
        gl_pca = global_loss_(X, Y_pca)
        gl_emb = global_loss_(X, Y)
        return np.exp(-(gl_emb-gl_pca)/gl_pca)
        

