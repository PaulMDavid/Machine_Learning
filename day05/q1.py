#!/usr/bin/env python
import matplotlib.pyplot as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
f=pd.read_csv("/home/ai11/Desktop/common/ML/Day5/ML_Assignments/ex1.txt")
X=f.as_matrix()
pl.xlabel="popu"
pl.ylabel="Profs"
pl.scatter(X[:,0],X[:,1])
pl.show()
y=X[:,1]
X=X[:,0]
X=X.reshape(-1,1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
lr=LinearRegression()
lr.fit(X_train,y_train)
p=lr.predict(input("enter population in 10000s"))
print p


