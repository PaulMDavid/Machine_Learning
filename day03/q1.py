#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
dataset=pd.read_csv("/home/ai11/Desktop/common/ML/Day3/Advertising.csv")
y=dataset.pop('sales')
X=dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
lr=LinearRegression()
lr.fit(X_train,y_train)
p=lr.predict(X_test)
plt.scatter(y_test,p)
plt.plot(y_test,p)
plt.show()

