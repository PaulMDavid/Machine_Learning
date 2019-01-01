#!/usr/bin/env python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
X=pd.read_csv("/home/ai11/nos.csv",delimiter='\s+')
X=X.as_matrix()
kmeans=KMeans(n_clusters=8)
kmeans.fit(X)
y_kmeans=kmeans.predict(X)

plt.scatter(X[:,0],X[:,1],s=25)
plt.show()

plt.scatter(X[:,0],X[:,1],s=50,c=y_kmeans,cmap='viridis')
plt.show()

