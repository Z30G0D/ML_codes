import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_sex(data):
    survived = data[data['Survived'] == 1]['Sex'].value_counts()
    not_survived = data[data['Survived'] == 0]['Sex'].value_counts()
    df = pd.DataFrame([survived, not_survived])
    df.index = ['Survived', 'Not survived']
    df.plot(kind='barh', stacked=True, figsize=[15, 9])
    plt.show()
    #data.plot.bar(x='Sex', y='Survived')
    plt.show()
    return 0


def plot_embarked(data):
    survived = data[data['Survived'] == 1]['Embarked'].value_counts()
    not_survived = data[data['Survived'] == 0]['Embarked'].value_counts()
    df = pd.DataFrame([survived, not_survived])
    df.index = ['Survived', 'Not Survived']
    df.plot(kind='barh', stacked=True, figsize=[15, 9])
    plt.show()
    return 0


def plot_age(data):
    figure = plt.figure(figsize=(15, 8))
    plt.hist([data[data['Survived'] == 1]['Age'], data[data['Survived'] == 0]['Age']], stacked=True, color=['b', 'r'],
             bins=30)
    plt.show()
    #print(survived)

    return 0


data = pd.read_csv('train.csv')
data1 = data
y = np.matrix(data['Survived'])
# filling missing ages
data['Age'].fillna(data['Age'].median(), inplace=True)
# Plotting Survival as a function of sex
#plot_sex(data)
# plotting survival as a function of embarking port
#plot_embarked(data)
# plotting survival as function og age
