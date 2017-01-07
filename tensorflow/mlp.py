#!/usr/bin/env python

from tensorflow.examples.tutorials.mnist import input_data
mnist =  input_data.read_data_sets('MNIST_data',one_hot = True)

import tensorflow as tf

learning_rate = 0.001
training_epoch = 15
batch_size = 100
display_step = 1

