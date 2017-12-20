import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('train.csv')
survived = data['Survived']

print (survived)
#data.plot.bar(x='Sex', y='Survived')
#plt.show()