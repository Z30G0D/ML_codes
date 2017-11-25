import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize

# python exercise 4 of andrew ng ml course by tomer nahshon


data = loadmat('/home/tomer/PycharmProjects/Machine learning andrew/machine-learning-ex4/ex4/ex4data1.mat')

X = data['X']
y = data['y']

# turning the y vector (labels) into appropriate output for the network using sklearn


encoder = OneHotEncoder(sparse=False)
y_encoder = encoder.fit_transform(y)


# calculating cost function, according to ex4.pdf, the codes used here are attached to the original zip
# file provided by coursera


# sigmoid function definition


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_propagate(X, theta1, theta2):
    # number of examples
    m = X.shape[0]
    # setting a1 as input to network(adding bias)
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    # using weights
    z2 = a1 * theta1.T
    # calculating a2 as input and adding bias
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    # calculating z3
    z3 = a2 * theta2.T
    # calculating hypothesis
    h = sigmoid(z3)

    return a1, z2, a2, z3, h


def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    # vector in the size of all of the weights - taken from different code by jdwett (credits in repository)
    # compute cost with  regularization
    m = X.shape[0]
    # shape input and output as matrices for cost calculation
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array(initial theta) into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # calculating feedfoward results by sigmoid functions
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # delta terms for back propagation
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    # compute cost using the log formula
    Jcost = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        Jcost += np.sum(first_term - second_term)

    Jcost = Jcost / m
    # Adding regularization
    theta1square = np.square(theta1)
    theta2square = np.square(theta2)
    Reg = (learning_rate * (theta1square.sum() + theta2square.sum())) / (2 * m)
    # Summing terms
    Jcost = Jcost + Reg

    # doing back propagation
    for t in range(m):
        # defining a, z, h, y for every example picture (total of 5000 pictures, 400 pixels each)
        a1t = a1[t, :]
        z2t = z2[t, :]
        a2t = a2[t, :]
        ht = h[t, :]
        yt = y[t, :]

        # output layer delta term
        d3t = ht - yt  # (1, 10)

        # hidden layer delta term (
        # bias
        z2t = np.insert(z2t, 0, values=np.ones(1))
        # here we need the weighted delta term using d3t - formula taken from figure 3
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
    # step 4  for backpropagation from the pdf
    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term (excluding the bias)
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m

    # calculating gradient (taken from jwdett code)
    # creates a long array of all the deltas found
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return Jcost, grad


# function for sigmoid gradient for backpropagation (section 2.1 in exercise 4)


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


# defining network parameters

# 20 by 20 pixels images reshaped to 400 size array
input_size = 400
# hidden layer size
hidden_size = 25
# outputlayer size
num_labels = 10
# arbitrary LR
learning_rate = 1

params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

result, gradient = backprop(params, input_size, hidden_size, num_labels, X, y_encoder, learning_rate)

print (result)
print (4 * "\n")
print(gradient.shape)

# using scipi for minimzing the cost function
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_encoder, learning_rate),
                method='TNC', jac=True, options={'maxiter': 10})
print(fmin)

