import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd


def principle_component(data):
    # Feature normalization
    m = np.size(data)
    data = (data-data.mean())/data.std()
    # calculating Covariance Matrix (nxn), where n is the number of features
    covmatrix = (np.dot(data.T, data)) / m
    # eigen vectors and values
    U, S, V = np.linalg.svd(covmatrix)
    return U, S, V
def reduce_dimension(data, U, k):
    # Reduce dimension, calculating Z
    U_reduced = U[:, :k]
    return np.dot(data, U_reduced)


def restored(z, U, k):
    # This function will calculate the restored data after dimension reduction
    U_reduced = U[:, :k]
    return np.dot(z, U_reduced.T)



data = loadmat('ex7/ex7data1.mat')
data = data['X']
# move to pandas format for plotting
data = pd.DataFrame(data, columns=['x1', 'x2'])
#fig, ax = plt.subplots(figsize=(12, 8))
#ax.scatter(data['x1'], data['x2'], color='r', label='Samples')
#ax.legend()
#plt.show()

# return to matrix format
data = data.values
U, S, V = principle_component(data)
#print(U)
z = reduce_dimension(data, U, 1)
x_new = restored(z, U, 1)
fig, ax = plt.subplots()
ax.scatter(x_new[:, 0], x_new[:, 1])
plt.show()

