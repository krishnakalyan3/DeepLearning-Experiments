#!/usr/bin/env python

from __future__ import print_function
import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

nb_classes = 10
img_rows, img_cols = 28,28

(X_train, y_train) , (X_test, y_test) = mnist.load_data()

