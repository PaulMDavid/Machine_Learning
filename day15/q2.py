from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
df=pd.read_csv("/home/ai11/Desktop/common/ML/Day15/sonar.csv",header=None)
y= df[60].reshape(-1,1)
print type(y)
X=df[list(range(0,60))]
print y
for l in range(0,len(y)):
 if y[l]=='R':
  y[l]=0
 else:
  y[l]=1
  
print y
encoder= OneHotEncoder()
y=encoder.fit_transform(y)
model=Sequential()
model.add(Dense(60,input_shape=(60,),activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(2,activation='softmax'))
optimizer=Adam(lr=0.001)
print(model.summary())
model.compile(optimizer,loss='mean_squared_error',metrics=['accuracy'])
model.fit(X,y,batch_size=5,verbose=2,epochs=10)
results=model.evaluate(X,y)
print(results[0])
print(results[1])

