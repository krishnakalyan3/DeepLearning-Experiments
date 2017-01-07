#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_o):
	h = tf.nn.sigmoid(tf.matmul(X,w_h))
	return tf.matmul(h,w_o)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder('float', [None, 784])
Y = tf.placeholder('float', [None, 10])

w_h = init_weights([784, 625])
w_o = init_weights([625,10])

py_x = model(X, w_h, w_o)

cost =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op = tf.argmax(py_x,1)

batch = 100

sess = tf.Session()
tf.initialize_all_variables().run(session=sess)


	
		
