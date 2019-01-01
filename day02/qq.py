#!/usr/bin/env python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
iris=load_iris()
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
l1=[]
#for i in range(1,25):
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
print(X_train.shape)
p=knn.predict(X_test)
# l1.append(p)
print y_test
print p
print(confusion_matrix(y_test,p))
print(accuracy_score(y_test,p))
print(knn)




