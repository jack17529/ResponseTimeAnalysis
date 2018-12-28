# -*- coding: utf-8 -*-
#SVR

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt

#Importing the dataset
"""dataset = pd.read_csv() #name.
X=dataset.iloc[:, 1:2].values
y=dataset.iloc[:,2].values"""


file='BTP Response Time Sheet.xlsx'
data = pd.ExcelFile(file)

data = data.parse('Sheet1')

data = data.fillna(value=0)

urls = range(len(data['Paths']))

X = np.array(data[data.keys()[2:4]].dropna())
y = np.array(data[data.keys()[-1]].dropna())


#Splitting the dataset into Testing Set and Training Set
train_len = int(len(X)*0.8)

X_train, X_test = X[:train_len], X[train_len:]
y_train, y_test = y[:train_len], y[train_len:]
urls_train, urls_test = urls[:train_len], urls[train_len:]

#Feature Scaling
#You don't need feature scaling in regression because it's an affine map.

#Fitting SVR to the dataset

regressor=SVR(kernel='rbf')
regressor.fit(X_train,y_train)

#Predicting a new result
y_pred = regressor.predict(X_test) #value

#Calculating Root Mean Squared Error(RMSE).

#initial_rms = sqrt(mean_squared_error(y_test, y_pred))
#print(initial_rms)

"""
# Do repeated K Fold Cross Validation.
from sklearn.model_selection import RepeatedKFold
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None) 

trmse=0
t=0

for train_index, test_index in rkf.split(X):
      #print("Train:", train_index, "Validation:",test_index)
      X_train, X_test = X[train_index], X[test_index] 
      y_train, y_test = y[train_index], y[test_index]
      
      #Fitting SVR to the dataset
      regressor=SVR(kernel='rbf')
      regressor.fit(X_train,y_train)

      #Predicting a new result
      y_pred = regressor.predict(X_test) #value

      #Calculating Root Mean Squared Error(RMSE).
      rms = sqrt(mean_squared_error(y_test, y_pred))
      #print(rms)
      trmse+=rms
      t+=1

print("So average Root Mean Square Error Of the Model Is ")
print(trmse/t)

"""

#Visualising the SVR results
plt.scatter(urls_test,y_test,color='red')
plt.plot(urls_test,regressor.predict(X_test),color='blue')
plt.title('SVR Prediciton')
plt.xlabel('urls')
plt.ylabel('time(ms)')
plt.show()
