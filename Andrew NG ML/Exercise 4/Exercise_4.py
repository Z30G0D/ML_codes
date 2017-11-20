import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# python exercise 4 of andrew ng ml course by tomer nahshon


data = loadmat('/home/tomer/PycharmProjects/Machine learning andrew/machine-learning-ex4/ex4/ex4data1.mat')

X = data['X']
y = data['y']

# turning the y vector (labels) into appropriate output for the network using sklearn

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_encoder = encoder.fit_transform(y)


# calculating cost function, according to ex4.pdf, the codes used here are attached to the original zip file provided by coursera


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


def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    # vector in the size of all of the weights - taken from different code by jdwett (credits in repository
    # compute cost with **NO** regularization
    m = X.shape[0]
    # shape input and output as matrices for cost calculation
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array(initial theta) into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # calculating feedfoward results by sigmoid functions
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # compute cost using the log formula
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    return J

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

result=cost(params, input_size, hidden_size, num_labels, X, y_encoder, learning_rate)

print (result)