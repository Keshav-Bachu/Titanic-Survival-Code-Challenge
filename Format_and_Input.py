# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 09:16:41 2018

@author: Keshav Bachu
"""

import pandas as pd
import TitanicModel as TM
import numpy as np

data = pd.read_csv('train.csv')  
data['Sex'] = data['Sex'].map({'female': 1, 'male': 0})
data['Embarked'] = data['Embarked'].map({'Q': 1, 'S': 0, 'C':2})
data = data.fillna(-1)


dataStripped = data.drop(columns = ['Name', 'Cabin', 'Ticket', 'PassengerId'])
YData = dataStripped['Survived']
XData = dataStripped
XData = XData.drop(columns = ['Survived'])

Ydata = YData.values
Xdata = XData.values

Ydata = Ydata.reshape(1, Ydata.shape[0])
Xdata = Xdata.T

networkShape = [64, 64, 32, 32, 64, 16, 16, 8, 1]
a, b = TM.trainModel(Xdata, Ydata, networkShape, itterations= 30000, learning_rate=0.00004, weightsExist=a)


Xtest = pd.read_csv('test.csv')
Xtest = Xtest.drop(columns = ['Name', 'Cabin', 'Ticket', 'PassengerId'])
Xtest['Sex'] = Xtest['Sex'].map({'female': 1, 'male': 0})
Xtest['Embarked'] = Xtest['Embarked'].map({'Q': 1, 'S': 0, 'C':2})

Ytest = pd.read_csv('gender_submission.csv')
Ytest = Ytest.drop(columns = ['PassengerId'])

Xtest = Xtest.values
Ytest = Ytest.values

Ytest = Ytest.reshape(1, Ytest.shape[0])
Xtest = Xtest.T

TM.predictor(a, networkShape, Xtest, Ytest)