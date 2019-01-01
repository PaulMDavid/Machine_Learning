#!/usr/bin/env python
from sklearn.datasets.samples_generator import make_moons
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
X,y_true=make_moons(300,noise=0.05)
clustering=SpectralClustering(n_clusters=2,affinity="nearest_neighbors")
label=clustering.fit_predict(X)
plt.scatter(X[:,0],X[:,1],s=25)
plt.show()

plt.scatter(X[:,0],X[:,1],s=50,c=label,cmap='viridis')
plt.show()

