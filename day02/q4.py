#!/usr/bin/env python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
imm=pd.read_csv("/home/ai11/ml/day02/Absenteeism_at_work.csv",delimiter=';')
target=(imm.pop('Absenteeism time in hours'))
print(imm)

X_train,X_test,y_train,y_test=train_test_split(imm,target,test_size=0.2)
l1=[]
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
p=knn.predict(X_test)
for i in range(1,25):
 knn=KNeighborsClassifier(n_neighbors=i)
 knn.fit(X_train,y_train)
 p=knn.predict(X_test)
 print(accuracy_score(y_test,p))
 l1.append(accuracy_score(y_test,p)) 
print(X_train.shape)
p=knn.predict(X_test)
# l1.append(p)
print y_test
print p
print(confusion_matrix(y_test,p))
print(accuracy_score(y_test,p))
plt.plot(range(1,25),l1)
plt.show()
#print(knn)




