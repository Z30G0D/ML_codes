import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('train.csv')
#
survived = data[data['Survived'] == 1]['Sex'].value_counts()
not_survived = data[data['Survived'] == 0]['Sex'].value_counts()
df = pd.DataFrame([survived, not_survived])
df.index = ['Survived', 'Not survived']
df.plot(kind='barh', stacked=True, figsize=[15, 9])
plt.show()
#data.plot.bar(x='Sex', y='Survived')
plt.show()