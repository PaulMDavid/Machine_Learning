#!/usr/bin/env python
from sklearn.datasets import load_boston
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
boston=load_boston()
X=boston.data
y=boston.target
print y
mlp=MLPRegressor(random_state=7,max_iter=1000,hidden_layer_sizes=1000,activation='relu',verbose=True,learning_rate='adaptive',alpha=0.000001)
print mlp.alpha
mlp.fit(X,y)
p=mlp.predict(X)
a=mean_squared_error(y,p)
print 'rmse=',a**0.5
b=mean_absolute_error(y,p)
print 'mae=',b
#plt.figure(figsize=(20,10))
plt.scatter(y,p)
plt.show()


