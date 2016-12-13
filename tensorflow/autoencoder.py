#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784

X = tf.placeholder('float', [None, n_input])

weights = {
	'eh1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'eh2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'dh1' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
	'dh2' : tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

biases = {
	'eb1': tf.Variable(tf.random_normal([n_hidden_1])),
	'eb2': tf.Variable(tf.random_normal([n_hidden_2])),
	'db1': tf.Variable(tf.random_normal([n_hidden_1])),
	'db2': tf.Variable(tf.random_normal([n_input])),
	
}


def encoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['eh1']), biases['eb1']))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['eh2']), biases['eb2']))
	return layer_2

def decoder(x):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['dh1']), biases['db1']))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['dh2']), biases['db2']))
	return layer_2

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
y_pred = decoder_op
y_true = X
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
total_batch = int(mnist.train.num_examples/batch_size)
for epoch in range(training_epochs):
	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		_,c = sess.run([optimizer, cost], feed_dict = {X: batch_xs})
	if epoch % display_step == 0:
		print "Epoch ", (epoch + 1) ,  "cost ", c
print "Optimization Finished"

encode_decode = sess.run(
	y_pred, feed_dict = {X:mnist.test.images[:examples_to_show]}
)
f,a = plt.subplots(2,10, figsize=(10,2))
for i in range(examples_to_show):
	a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
	a[1][i].imshow(np.reshape(encode_decode[i],(28,28)))
	plt.savefig('auto.png')
