import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy import stats

def estimate_gaussian(X):
    """Estimating gaussian for probability plotting"""
    m = X.shape[0]
    n = X.shape[1]
    mu = sum(X) / m
    sigma2 = sum(np.square(X - mu)) / m
    return mu, sigma2


def select_Threshold(p, yval):
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
    stepsize = (p.max() - p.min()) / 1000
    for epsilon in np.arange(p.min()+ stepsize, p.max(), stepsize):
        pred_anomaly = p < epsilon
        # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
        TP = np.sum(np.logical_and(pred_anomaly == 1, yval == 1)).astype(float)

        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        TN = np.sum(np.logical_and(pred_anomaly == 0, yval == 0)).astype(float)

        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        FP = np.sum(np.logical_and(pred_anomaly == 1, yval == 0)).astype(float)

        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        FN = np.sum(np.logical_and(pred_anomaly == 0, yval == 1)).astype(float)

        prec = TP / (TP + FP)

        rec = TP / (TP + FN)

        F1 = (2 * prec * rec) / (prec + rec)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1

data = loadmat('machine-learning-ex8/ex8/ex8data1.mat')
X = data['Xval']
Y = data['yval']
gauss_mu, gauss_sigma2 = estimate_gaussian(X)
mu = gauss_mu[0]
# calculating probability for each sample from the CV set
pval = stats.norm(gauss_mu, gauss_sigma2).pdf(X)

epsilon, F1 = select_Threshold(pval, Y)
print('Best epsilon:')
print(epsilon)
print('Best F1:')
print(F1)
outliers = np.where(pval < epsilon)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:, 0], X[:, 1])
ax.scatter(X[outliers[0], 0], X[outliers[0], 1], s=50, color='r', marker='o')
plt.show()
