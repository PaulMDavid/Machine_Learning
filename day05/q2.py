#!/usr/bin/env python
import matplotlib.pyplot as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
f=(pd.read_csv("/home/ai11/Desktop/common/ML/Day5/ML_Assignments/ex2.txt"))
f=f.as_matrix()
X=(f[1:,[0,1]])
y=(f[1:,2])
X[:,1]=(X[:,1])*1000
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
lr=LinearRegression()
lr.fit(X_train,y_train)
p=lr.predict(X_test)
pl.scatter(p, y_test)
pl.show()


