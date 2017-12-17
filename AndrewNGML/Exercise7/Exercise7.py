# This exercise includes implementation of k-means and PCA algorithms according to andrewng course in coursera
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def compute_centroids(X, k, idx):
    m, n = X.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        # finding the correlated point for the current centroid
        indices = np.where(idx == i)
        # calculating the new centroid location according to the correlated examples
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()

    return centroids


def runKmeans(X, initial_centroids, iters):
    centroids = initial_centroids
    m, n = X.shape
    k = initial_centroids.shape[0]
    # initialize which example belongs to which centroid
    idx = np.zeros(m)
    for i in range(iters):
        idx = closest_centroids(X, centroids)
        centroids = compute_centroids(X, k, idx)
    return idx, centroids
def init_centroids(X, k):
    # permutate the samples
    rdidx = np.random.permutation(np.size(X[:, 0]))
    # use the permutation to establish the random centroids
    init = mydata[rdidx[0:k], :]
    return init


mydata = loadmat('ex7/ex7data2.mat')
mydata = mydata['X']
# Three initial centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
iters = 10

#print (idx)
# output is  [0 2 1] for the first 3 samples and is equivalent to [1 3 2] for the matlab implementation


idx, centroids = runKmeans(mydata, init_centroids(mydata, 3), iters)

#print(idx)
#print(centroids)

first_cluster_points = np.where(idx == 0)[0]
first_cluster = mydata[first_cluster_points, :]
second_cluster_points = np.where(idx == 1)[0]
second_cluster = mydata[second_cluster_points, :]
third_cluster_points = np.where(idx == 2)[0]
third_cluster = mydata[third_cluster_points, :]
plt.scatter(first_cluster[:, 0], first_cluster[:, 1], color='red')
plt.scatter(second_cluster[:, 0], second_cluster[:, 1], color='green')
plt.scatter(third_cluster[:, 0], third_cluster[:, 1], color='purple')
plt.show()



