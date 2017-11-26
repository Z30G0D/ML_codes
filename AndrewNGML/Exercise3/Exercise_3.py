import numpy as np
from scipy.io import loadmat

data = loadmat("ex3data1.mat")


def sigmoid(z):
    return 1/(1+np.exp(-z))


def cost(theta,X, y, lamb):
    # Avoiding loops , vectorized approach
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    # first term including y=1 classes
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    # second term includes y=0 classes
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    # reg term to avoid overfitting
    reg = (lamb / 2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    # concluding cost
    j = + reg + np.sum(first - second) / (len(X))
    return j



