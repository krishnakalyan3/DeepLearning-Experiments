#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

rng = np.random
learning_rate = 0.01
epoch = 1000
display_step = 50

train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n = train_X.shape[0]
X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

pred = tf.add(tf.mul(X,W), b)
cost = tf.reduce_sum(tf.pow(pred - Y, 2))/(2*n)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for epoch in range(epoch):
	for(x,y) in zip(train_X,train_Y):
		sess.run(optimizer, feed_dict={X:x,Y:y})
	if (epoch + 1) % display_step == 0:
		c = sess.run(cost, feed_dict = {X:train_X,Y:train_Y })
		print "Epoch ", (epoch + 1), "cost ", c, "W ", sess.run(W), "b ", sess.run(b)
print "Optimization finished"
training_cost = sess.run(cost, feed_dict= {X: train_X, Y : train_Y})
print "Train Cost ", training_cost, "W ", sess.run(W), "b = ", sess.run(b)
print "\n"

plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
plt.legend()
plt.savefig('lm.png')
