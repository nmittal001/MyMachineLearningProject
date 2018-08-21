#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 16:37:37 2017

@author: root
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('train.csv')
X=dataset.iloc[:,[2,4,5]].values
Y=dataset.iloc[:,1].values

dataset1=pd.read_csv('test.csv')
M=dataset1.iloc[:,[1,3,4]].values
#N=dataset.iloc[:,1].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X= LabelEncoder()
X[:,1]=labelencoder_X.fit_transform(X[:,1])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_M= LabelEncoder()
M[:,1]=labelencoder_M.fit_transform(M[:,1])

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, [0,2]])
X[:, [0,2]] = imputer.transform(X[:, [0,2]])

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(M[:, [0,2]])
M[:, [0,2]] = imputer.transform(M[:, [0,2]])



'''

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X, Y)
'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X,Y)
# Predicting the Test set results
y_pred = classifier.predict(M)