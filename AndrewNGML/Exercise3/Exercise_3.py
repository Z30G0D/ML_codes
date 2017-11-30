import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

# for debugging - seeing entire array
#np.set_printoptions(threshold=np.inf)

data = loadmat("ex3data1.mat")


def sigmoid(z):
    """sigmoid function"""
    return 1/(1+np.exp(-z))


def cost(theta, X, y, lamb):
    """"computing the cost function according to logistic regression including regularization term"""
    # Avoiding loops , vectorized approach
    X = np.matrix(X)
    m = len(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    # first term including y=1 classes
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    # second term includes y=0 classes
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    # reg term to avoid overfitting - excluding theta(0)
    reg = (lamb / 2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    # concluding cost
    j = reg + np.sum(first - second) / m
    return j


def gradients(X, y, theta, lamb):
    """Calculating gradients (derviatives) for updating the parameters"""
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    m = len(X)
    # counting number of parameters
    param = theta.ravel().shape[1]
    # for looping
    param = int(param)

    grads = np.zeros(param)
    z = X * theta.T
    # error vector
    error = sigmoid(z) - y
    # calculating first term(intercept parameter) with *no* regularization to avoid penalizing all parameters
    first_term = np.multiply(error, X[:, 0])
    grads[0] = np.sum(first_term) / m
    for k in range(param):
        all_terms = np.multiply(error, X[:, k])
        if k == 0:
            break
        else:
            grads[k] = (np.sum(all_terms) / m) + ((lamb / m) * theta[:, k])
    return grads.T


def onevsall(X, y, num_labels, lamb):

    # number of columns (features)
    params = X.shape[1]
    # creating a theta matrix (rows as number of calsses , columns as number of parameters)
    theta_vector = np.zeros((num_labels, params + 1))

    # number of rows (examples)
    rows = X.shape[0]

    # according to the pdf exercise , insert the bias term (intercept)
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # For one vs all we go through each class and classify it as 1, and all other as 0. so y is a vector of
    for i in range(1, num_labels + 1):
        # initialize theta for minimizing function
        theta = np.zeros(params + 1).T
        # creating a y specific for our i category
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function -taken from scipy documentation
        # need to debug this one
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, lamb), method='TNC', jac=gradients)
        theta_vector[i - 1, :] = fmin.x

    return theta_vector

# note to self: checking sizes of matrices called by fmin
# training the algorithm
theta_final = onevsall(data['X'], data['y'], 10, 1)


