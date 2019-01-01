#!/usr/bin/env python
import matplotlib.pyplot as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
f=(pd.read_csv("/home/ai11/Desktop/common/ML/Day5/ML_Assignments/titanic.csv"))
f.dropna(inplace=True)
f=f.as_matrix()
le=preprocessing.LabelEncoder()
le.fit(f[:,4])
f[:,4]=le.transform(f[:,4])
#print f[:,4]
X=f[1:,[2,4,5]]
X1=f[1:,]
X2=f[1:,1]
y=f[1:,1]
y=y.astype('int')
#pl.scatter(X1,X2,c=y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
lr=LogisticRegression()
lr.fit(X_train,y_train)
p=lr.predict(X_test)
pl.scatter(p, y_test)
pl.show()
print p
