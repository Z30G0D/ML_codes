import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pylab
from sklearn import svm
from scipy.io import loadmat

matrix = loadmat('Data/ex6data1.mat')

#print(matrix)
# Displaying data visually in matplot lib


# convert to pandas dataframe for plotting
# features table
mydata = pd.DataFrame(matrix['X'], columns=['x1', 'x2'])
# tags table
mydata['y'] = matrix['y']

#print(mydata['y'])

# boolean arrays for plotting(redundant?)
pos = mydata[mydata['y'].isin([1])]
neg = mydata[mydata['y'].isin([0])]



# plot scatter
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(pos['x1'], pos['x2'], s=50, color='r', label='Positive')
ax.scatter(neg['x1'], neg['x2'], s=50, color='g', label='Negatove')
ax.legend()
plt.show()


# pylab.show()

# create support vector machines algorithm using
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
svc.fit(mydata[['x1', 'x2']], mydata['y'])
svm1 = svc.score(mydata[['x1', 'x2']], mydata['y'])
print ("SVM algorithm with C=1 is equal to {} ".format(svm1))


# svm using larger C to avoid underfitting
svc2 = svm.LinearSVC(C=10, loss='hinge', max_iter=1000)
svc2.fit(mydata[['x1', 'x2']], mydata['y'])
svm2 = svc2.score(mydata[['x1', 'x2']], mydata['y'])
print ("SVM algorithm with C=1000 is equal to {} ".format(svm2))




