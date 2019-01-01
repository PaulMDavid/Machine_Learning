#!/usr/bin/env python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
iris=load_iris()
X=iris.data
y=iris.target
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
p=knn.predict([[3,5,4,2]])
print p




