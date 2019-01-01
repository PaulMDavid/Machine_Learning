#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
dataset=pd.read_csv("/home/ai11/lottery.csv",delimiter='\s+')
dataset=dataset.as_matrix()
y=dataset[:,-1]
X=dataset[:,[0,1,2]]
#print y
#print X
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
lr=LinearRegression()
lr.fit(X_train,y_train)
p=lr.predict(X_test)
print(mean_squared_error(y_test,p))
plt.scatter(y_test,p)
#plt.plot(y_test,p)
plt.show()

