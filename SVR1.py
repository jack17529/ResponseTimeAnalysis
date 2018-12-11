# -*- coding: utf-8 -*-
#SVR

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#Importing the dataset
"""dataset = pd.read_csv() #name.
X=dataset.iloc[:, 1:2].values
y=dataset.iloc[:,2].values"""

file='BTP Response Time Sheet.xlsx'
data = pd.ExcelFile(file)

data = data.parse('Sheet1')
data = data.fillna(value=0)

paths = range(len(data['Paths']))

X = np.array(data[data.keys()[2:4]].dropna())
y = np.array(data[data.keys()[-1]].dropna())

#Splitting the dataset into Testing Set and Training Set
train_len = int(len(X)*0.8)

X_train, X_test = X[:train_len], X[train_len:]
y_train, y_test = y[:train_len], y[train_len:]
paths_train, paths_test = paths[:train_len], paths[train_len:]

#Feature Scaling

#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X_train,y_train)

#Predicting a new result
y_pred = regressor.predict(X_test) #value

#Visualising the SVR results
plt.scatter(paths_test,y_test,color='red')
plt.plot(paths_test,regressor.predict(X_test),color='blue')
plt.title('SVR Prediciton')
plt.xlabel('paths')
plt.ylabel('time(ms)')
plt.show()