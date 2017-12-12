# This exercise includes implementation of k-means and PCA algorithms according to andrewng course in coursera
from scipy.io import loadmat
import numpy as np


def closest_centroids(X, Centroids):
    length = np.size(X, 0)
    k = np.size(Centroids, 0)
    idx = np.zeros((length,), dtype=np.int)
    for i in range(length):
        final_dist = 1000000
        for j in range(k):
            # calculating distance for c(j) centroid mean
            dist = np.sum(np.square(X[i, :] - Centroids[j, :]))
            # choosing the shortest distance
            if dist < final_dist:
                final_dist = dist
                idx[i] = j
    return idx
def compute_centroids(X, centroids ,idx):



mydata = loadmat('ex7/ex7data2.mat')
mydata = mydata['X']
# Three initial centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = closest_centroids(mydata, initial_centroids)

print (idx)
# output is  [0 2 1] for the first 3 samples and is equivalent to [1 3 2] for the matlab implementation

