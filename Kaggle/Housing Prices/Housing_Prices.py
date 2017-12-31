import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize


def lot_visualization(data):
    data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].mean())
    df = data[['LotFrontage', 'SalePrice']]
    # print(df)
    # df.index = ["Linear feet of street connected to property", "Selling Price"]
    df = df.sort_values(['LotFrontage'])
    df.plot(x='LotFrontage', y='SalePrice')
    plt.show()
    return 0

train = pd.read_csv('train.csv')
print(train.SalePrice.skew)
test = pd.read_csv('test.csv')

a = lot_visualization(train)
