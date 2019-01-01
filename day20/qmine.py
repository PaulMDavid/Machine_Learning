#!/usr/env/python
import tensorflow as tf
import numpy as np
data=np.loadtxt('/home/ai11/Desktop/common/ML/Day20/pimaindians.csv',delimiter=',')
print data

learning_rate = 0.001
training_epochs = 2
batch_size = 10
display_step = 1
X=data[:,0:8]
y=data[:,8]
print X.shape

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = i # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

X=tf.placeholder(dtype='float32',size=[None,8])
y=tf.placeholder(dtype='int',size=[None,1])


