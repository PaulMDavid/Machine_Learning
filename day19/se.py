#!/usr/bin/env python
import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

sess = tf.Session()
print (sess.run(c*c))
sess.close()

