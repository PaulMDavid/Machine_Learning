#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
dataset=pd.read_csv("/home/ai11/Desktop/common/ML/Day3/pimaindians.csv")
dataset=dataset.as_matrix()
X=dataset[:,[0,1,2,3,4,5,6,7,8]]
y=dataset[:,-1]
print y
print X
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
lr=LogisticRegression()
lr.fit(X_train,y_train)
p=lr.predict(X_test)
print(confusion_matrix(y_test,p))
plt.scatter(y_test,p)
plt.plot(y_test,p)
plt.show()

