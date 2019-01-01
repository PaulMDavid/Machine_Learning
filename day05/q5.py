#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
f=(pd.read_csv("/home/ai11/Desktop/common/ML/Day5/ML_Assignments/mining.csv"))
f.dropna(inplace=True)
f=f.as_matrix()
le=preprocessing.LabelEncoder()
le.fit(f[1:,2])
f[1:,2]=le.transform(f[1:,2])
le=preprocessing.LabelEncoder()
le.fit(f[1:,1])
f[1:,1]=le.transform(f[1:,1])
#print f[:,4]
X=f[1:,range(1,10)]
X1=f[1:,]
X2=f[1:,1]
y=f[1:,0]
y=y.astype('int')
#pl.scatter(X1,X2,c=y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#lr = linear_model.LogisticRegression()
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X_train,y_train)

#lr=LogisticRegression()
#lr.fit(X_train,y_train)
p=mul_lr.predict(X_test)
plt.scatter(p, y_test)
plt.show()
print p

#mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_x, train_y)
