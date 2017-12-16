import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat
import numpy as np
from scipy import misc


def init_centroids(X, k):
    # permutate the samples
    rdidx = np.random.permutation(np.size(X[:, 0]))
    # use the permutation to establish the random centroids
    init = X[rdidx[0:k], :]
    return init


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

# using different picture than described
img = misc.imread('ex7/bridge-24bit-png.png')
imgplot = plt.imshow(img)
plt.show()
print (type(img))
print(img.shape, img.dtype)



data = loadmat('ex7/bird_small.mat')
data = img
#data = data['A']
data = data / 255.
# Reshaping in order to get every point with
data_recovered = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]))

# Taken from exercise 7 solution
iters = 10
idx, centroids = runKmeans(data_recovered, init_centroids(data_recovered, 16), iters)
# convergence affirm
idx = closest_centroids(data_recovered, centroids)

New_picture = centroids[idx.astype(int), :]

# shape the array for the picture
New_picture = np.reshape(New_picture, (data.shape[0], data.shape[1], data.shape[2]))

plt.imshow(New_picture)
plt.show()

print (idx)
print('Centroids')
print(centroids)
