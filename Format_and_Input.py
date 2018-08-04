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
data['Age'] = data['Age'].fillna(27)
data = data.fillna(-1)


dataStripped = data.drop(columns = ['Name', 'Cabin', 'Ticket', 'PassengerId'])
YData = dataStripped['Survived']
XData = dataStripped
XData = XData.drop(columns = ['Survived'])

Ydata = YData.values
Xdata = XData.values

Ydata = Ydata.reshape(1, Ydata.shape[0])
Xdata = Xdata.T

Xtest = Xdata[:, 801:]
Ytest = Ydata[:, 801:]

Xdata = Xdata[: , :800]
Ydata = Ydata[:, :800]


networkShape = [16,16,16,16,1]
a, b = TM.trainModel(Xdata, Ydata, networkShape, itterations= 10000, learning_rate=0.004)


"""
Xtest = pd.read_csv('test.csv')
Xtest = Xtest.drop(columns = ['Name', 'Cabin', 'Ticket', 'PassengerId'])
Xtest['Sex'] = Xtest['Sex'].map({'female': 1, 'male': 0})
Xtest['Embarked'] = Xtest['Embarked'].map({'Q': 1, 'S': 0, 'C':2})
Xtest['Age'] = Xtest['Age'].fillna(27)
Xtest = Xtest.fillna(-1)

Ytest = pd.read_csv('gender_submission.csv')
Ytest = Ytest.drop(columns = ['PassengerId'])

Xtest = Xtest.values
Ytest = Ytest.values

Ytest = Ytest.reshape(1, Ytest.shape[0])
Xtest = Xtest.T
"""

#_, check = TM.predictor(a, networkShape, Xtest, Ytest)

"""
df = pd.DataFrame(check.T)
df = df.astype(int)
df.to_csv("file_path.csv")
"""