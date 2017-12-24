import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_sex(data):
    """plot sex as a function of survival"""
    survived = data[data['Survived'] == 1]['Sex'].value_counts()
    not_survived = data[data['Survived'] == 0]['Sex'].value_counts()
    df = pd.DataFrame([survived, not_survived])
    df.index = ['Survived', 'Not survived']
    df.plot(kind='barh', stacked=True, figsize=[15, 9])
    plt.show()
    #data.plot.bar(x='Sex', y='Survived')


def plot_age(data):
    """plotting age as a function of survival"""
    figure = plt.figure(figsize=(15, 8))
    plt.hist([data[data['Survived'] == 0]['Age'], data[data['Survived'] == 1]['Age']], stacked=True, color=['b', 'r'],
             bins=3, label=['Survived', 'Dead'])
    plt.legend()
    plt.show()
    #print(survived)


def plot_fare(data):
    """plotting fare as a function of survival"""
    plt.figure(figsize=(15, 8))
    plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']], stacked=True, bins=[0, 5, 10, 15, 20, 25, 30, 35], color=['y', 'g'], label=['Survived', 'Not Survived'])
    plt.legend(loc='best')
    plt.show()


def plot_siblings(data):
    """plotting siblings as a function of survival"""
    plt.figure(figsize=(15, 8))
    plt.hist([data[data['SibSp'] == 0]['Survived'], data[data['SibSp'] != 0]['Survived']], stacked=True, bins=2, color=['y', 'g'], label=['Survived', 'Not Survived'])
    plt.legend(loc='best')
    plt.show()


def age_vs_fare(data):
    plt.figure()
    ax = plt.subplot()
    ax.scatter(data[data['Survived'] == 1]['Age'], data[data['Survived'] == 1]['Fare'], s=50, color='b',
               label='Survived')
    ax.scatter(data[data['Survived'] == 0]['Age'], data[data['Survived'] == 0]['Fare'], s=50, color='r', label='Dead')
    plt.title('Age Vs. Fare')
    plt.xlabel('Age')
    plt.ylabel('Fare')
    plt.legend()
    plt.show()


def combined_data():
    """This function combines the test data and the training data"""
    data = pd.read_csv('train.csv')
    y = data['Survived']
    data1 = pd.read_csv('test.csv')
    combined = data.append(data1)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)
    return combined, y

combined_data()
