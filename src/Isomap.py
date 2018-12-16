from __future__ import division

from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path
from numpy import linalg as LA
import numpy as np
import math
import os
import KernelPCA

n_neighbors=8
n_components=2
eigen_solver='auto'
n_jobs=1

def Landmark_Isomap(D, ndims, landmarks, dist_matrix):

    ALL_matrix = D

    print ("Generate NN Graph...")
    if dist_matrix:
        dist_matrix_ = np.loadtxt(os.path.join(dist_matrix,"X_dist.txt"))
    else:
        nbrs_ = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=n_jobs)
        nbrs_.fit(ALL_matrix)
        kng = kneighbors_graph(nbrs_, n_neighbors, mode='distance', n_jobs=n_jobs)

        print ("Generate distance matrix of NN graph...")
        dist_matrix_ = graph_shortest_path(kng,
                                           method='auto',
                                           directed=False)
    G_D = dist_matrix_[landmarks, :]
    landmarks = np.array(landmarks)
    G_ = dist_matrix_[landmarks[:, None], landmarks]
  
    G = G_** 2
    G *= -0.5
    n = len(landmarks)
    N = len(D)
    """
    xy1 = KernelPCA(n_components=ndims,
              kernel="precomputed",
              eigen_solver='auto',
              tol=0, max_iter=None,
              n_jobs=n_jobs).fit_transform(G)
    """
    eigenxy, eigenval = KernelPCA.KernelPCA(n_components=ndims,
                                            kernel="precomputed",
                                            eigen_solver='auto',
                                            tol=0, max_iter=None,
                                            n_jobs=n_jobs).fit_transform(G)
    
    xy = eigenxy
    val = eigenval
  
    
    for i in range (0, ndims):
        xy[:, i] = xy[:, i]*np.sqrt(val[i])


    xy1 = np.zeros((len(D), ndims))
    LT = xy.transpose()

    for i in range (0, ndims):
        LT[i, :] = LT[i, :]/val[i]
    deltan = G.mean(0)
    
    for x in range (0, len(D)):
        deltax = G_D[:, x]
        xy1[x, :] = 1/2 * (LT.dot((deltan-deltax))).transpose()

    return xy1, xy1[landmarks]


def Tearing_Isomap(dist_matrix,ndims):

   
    G_ = dist_matrix 
    G = G_** 2
    G *= -0.5

    """
    xy1 = KernelPCA(n_components=ndims,
              kernel="precomputed",
              eigen_solver='auto',
              tol=0, max_iter=None,
              n_jobs=n_jobs).fit_transform(G)
    """
    eigenxy, eigenval = KernelPCA.KernelPCA(n_components=ndims,
                                            kernel="precomputed",
                                            eigen_solver='auto',
                                            tol=0, max_iter=None,
                                            n_jobs=n_jobs).fit_transform(G)
    
    xy = eigenxy
    val = eigenval
  
    
    for i in range (0, ndims):
        xy[:, i] = xy[:, i]*np.sqrt(val[i])

    return xy


