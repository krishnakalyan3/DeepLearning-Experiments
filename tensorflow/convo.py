#!/usr/bin/env python

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist =  input_data.read_data_sets('MNIST_data',one_hot = True)

learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

n_input = 784
n_classes = 10
dropout = 0.75

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W, b, strides=1):
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x,b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2):
	return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1],padding='SAME')

def conv_net(x, weights, biases, dropout):
	x = tf.reshape(x, shape=[-1,28,28,1])
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	conv1 = maxpool2d(conv1,k=2)
	conv2 = conv2d(conv1, weight['wc2'], biases['bc2'])
	conv2 = maxpool2d(conv2, k=2)
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	fc1 = tf.nn.dropout(fc1, dropout)
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out	

weigths = {
	'wc1': tf.Variable(tf.random_normal([5,5,1,32])),
	'wc2': tf.Variable(tf.random_normal([5,5,32,64])),
	'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
	'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
	'bc1': tf.Variable(tf.random_normal([32])),
	'bc2': tf.Variable(tf.random_normal([64])),
	'db1': tf.Variable(tf.random_normal([1024])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}	
