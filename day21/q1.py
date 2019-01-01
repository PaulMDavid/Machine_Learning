import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras import regularizers
resume_weights = "fashion_mnist-cnn-best.hdf5"
training_set=pd.read_csv('/home/ai11/Desktop/common/ML/Day21/BTC-USD.csv')  
h=training_set.iloc[:,1]
training_set2=training_set.iloc[:,1:6]        #selecting the second column
training_set1=training_set.iloc[:,1:2]        #selecting the second column
print(training_set2.head())                   #print first five rows
training_set1=training_set1.values    
#print training_set1.shape
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()           
xtra=training_set2[0:2764]                #scaling using normalisation 
training_set1 = sc.fit_transform(training_set1)
xtrain=training_set2[0:2764]               #input values of rows [0-2694
ytrain=training_set1[1:2765]  
print ytrain.shape
print type(ytrain)
xtrain=xtrain.as_matrix()
xtrain = sc.transform(xtrain)
xtrain = np.reshape(xtrain, (2764,1,5))     #Reshaping into required shape for Keras
#ytrain=ytrain.as_matrix()
#ytrain=np.reshape(ytrain,(1,2764,5))

#importing keras and its packages

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

regressor=Sequential()                                                      #initialize the RNN
filepath = "mnist-cnn-best.hdf5"
checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')

regressor.add(LSTM(units=1,activation='linear',bias_initializer='normal',input_shape=(1,5),bias_regularizer=regularizers.l2(0.01)))      #adding input layerand the LSTM layer 

if os.path.isfile(resume_weights):
        print ("Resumed model's weights from {}".format(resume_weights))
        # load weights
        model.load_weights(resume_weights)

#regressor.add(Dense(units=1))                                               #adding output layers
regressor.compile(optimizer='adam',loss='mean_squared_error') 
regressor.fit(xtrain,ytrain,batch_size=32,epochs=50,callbacks=[checkpoint])
test_set = pd.read_csv('/home/ai11/Desktop/common/ML/Day21/BTCtest.csv')
#print test_set.head()  


real_stock_price = test_set.iloc[:,1]
real_stock_price=real_stock_price.values

#print real_stock_price

#getting the predicted BTC value of the first week of Dec 2017  
inputs = xtra.as_matrix()
#inputs=inputs.reshape(1,-1)
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (2764, 1,5))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print predicted_stock_price.shape,xtra.shape
print h[2763],predicted_stock_price[2754]
plt.plot(h, color = 'green', label = 'Current BTC Value')
#plt.savefig('current.png')
plt.plot(predicted_stock_price, color = 'red', label = 'Predicted BTC Value')
plt.title('BTC Value Prediction')
plt.xlabel('Days')
plt.ylabel('BTC Value')
plt.savefig('predict.png')
plt.legend()
plt.show()
