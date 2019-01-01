import matplotlib.pyplot as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
f=(pd.read_csv("/home/ai11/Desktop/common/ML/Day5/ML_Assignments/ex3.txt"))
f=f.as_matrix()
X=f[1:,[0,1]]
X1=f[1:,0]
X2=f[1:,1]
y=f[1:,2]
pl.scatter(X1,X2,c=y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
lr=LinearRegression()
lr.fit(X_train,y_train)
p=lr.predict(X_test)
#pl.scatter(,c='red')
pl.show()

