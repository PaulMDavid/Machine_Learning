#!/usr/bin/env python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
imm=pd.read_csv("/home/ai11/Desktop/common/ML/Day2/Questions/Immunotherapy.csv")
target=(imm.pop('Result_of_Treatment'))
#print(imm)

X_train,X_test,y_train,y_test=train_test_split(imm,target,test_size=0.2)
l1=[]
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
p=knn.predict(X_test)
score=cross_val_score(knn,X_train,y_train,cv=10)
print('cross=',score)
l1.append(accuracy_score(y_test,p))
#print(X_train.shape)
p=knn.predict(X_test)
# l1.append(p)
print y_test
print p
print(confusion_matrix(y_test,p))
print(accuracy_score(y_test,p))
print(f1_score(y_test,p))
print(classification_report(y_test,p))
#plt.plot(range(1,25),l1)
#print(knn)
