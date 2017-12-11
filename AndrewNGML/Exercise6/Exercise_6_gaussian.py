import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pylab
from sklearn import svm
from scipy.io import loadmat
import numpy as np
mydata2 = loadmat('Data/ex6data2.mat')

def gaussian(xi, xj, sigma):
    """This function calculates the similarity between two examples (each one contains two features in this example"""
    result = np.exp(-(np.sum(np.square(xi - xj)) / (2 * np.square(sigma))))

    return result


first = np.array([1, 2])
second = np.array([1, 5])

print (gaussian(first, second, 1))


