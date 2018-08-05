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


networkShape = [16,8,1]
"""
a1, b1, c1 = TM.trainModel(Xdata[0, :].reshape(1, Xdata.shape[1]), Ydata, networkShape, itterations= 3000, learning_rate=0.001)
a2, b2, c2 = TM.trainModel(Xdata[1, :].reshape(1, Xdata.shape[1]), Ydata, networkShape, itterations= 3000, learning_rate=0.004)
a3, b3, c3 = TM.trainModel(Xdata[2, :].reshape(1, Xdata.shape[1]), Ydata, networkShape, itterations= 3000, learning_rate=0.004)
a4, b4, c4 = TM.trainModel(Xdata[3, :].reshape(1, Xdata.shape[1]), Ydata, networkShape, itterations= 3000, learning_rate=0.004)
a5, b5, c5 = TM.trainModel(Xdata[4, :].reshape(1, Xdata.shape[1]), Ydata, networkShape, itterations= 3000, learning_rate=0.004)
a6, b6, c6 = TM.trainModel(Xdata[5, :].reshape(1, Xdata.shape[1]), Ydata, networkShape, itterations= 3000, learning_rate=0.004)
a7, b7, c7 = TM.trainModel(Xdata[6, :].reshape(1, Xdata.shape[1]), Ydata, networkShape, itterations= 3000, learning_rate=0.004)
"""

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
_, check1, f1 = TM.predictor(a1, networkShape, Xtest[0, :].reshape(1, Xtest.shape[1]), Ytest)
_, check2, f2 = TM.predictor(a2, networkShape, Xtest[1, :].reshape(1, Xtest.shape[1]), Ytest)
_, check3, f3 = TM.predictor(a3, networkShape, Xtest[2, :].reshape(1, Xtest.shape[1]), Ytest)
_, check4, f4 = TM.predictor(a4, networkShape, Xtest[3, :].reshape(1, Xtest.shape[1]), Ytest)
_, check5, f5 = TM.predictor(a5, networkShape, Xtest[4, :].reshape(1, Xtest.shape[1]), Ytest)
_, check6, f6 = TM.predictor(a6, networkShape, Xtest[5, :].reshape(1, Xtest.shape[1]), Ytest)
_, check7, f7 = TM.predictor(a7, networkShape, Xtest[6, :].reshape(1, Xtest.shape[1]), Ytest)

"""
df = pd.DataFrame(check.T)
df = df.astype(int)
df.to_csv("file_path.csv")
"""