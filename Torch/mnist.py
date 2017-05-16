#!/usr/bin/env python3

from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from scipy.stats import itemfreq
import bcolz
import os
from sklearn.utils import shuffle

SEED = 500


def split_data(train_size=50000, val_size=10000):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #print(itemfreq(y_train).astype(int))

    X, y = shuffle(x_train, y_train)
    x_train, x_val, y_train, y_val = train_test_split(X, y, train_size=train_size,
                                test_size=val_size, stratify=y, random_state=SEED)

    save_np(TRAIN + str(SEED) + '/' + 'x_train.bc', x_train)
    save_np(TRAIN + str(SEED) + '/' + 'y_train.bc', y_train)
    save_np(TEST + str(SEED) + '/' + 'x_test.bc', x_test)
    save_np(TEST + str(SEED) + '/' + 'y_test.bc', y_test)
    save_np(VAL + str(SEED) + '/' + 'x_val.bc', x_val)
    save_np(VAL + str(SEED) + '/' + 'y_val.bc', y_val)

    return 'balanced'

def save_np(file_name, arr):
    c = bcolz.carray(arr, rootdir=file_name, mode='w')
    c.flush()

if __name__ == '__main__':
    TRAIN = '../../data/mnist/train/'
    TEST = '../../data/mnist/test/'
    VAL = '../../data/mnist/val/'

    if not os.path.exists(TRAIN + str(SEED)):
        os.makedirs(TRAIN + str(SEED))
    if not os.path.exists(TEST + str(SEED)):
        os.makedirs(TEST + str(SEED))
    if not os.path.exists(VAL + str(SEED)):
        os.makedirs(VAL + str(SEED))

    print(split_data())