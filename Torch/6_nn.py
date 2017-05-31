#!/usr/bin/env python3
# https://github.com/PythonWorkshop/Intro-to-TensorFlow-and-PyTorch/blob/master/PyTorch%20Tutorial.ipynb

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax
import torch.optim as optim

df = pd.read_csv('Data/winequality-red-cleaned.csv', sep=',')
y = pd.DataFrame([0. if item == 'Good' else 1. for item in df['category']])
X = df.drop(['quality', 'category'], axis=1)

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
learning_rate = 0.005

