#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F


num_classes = 10
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 128
learning_rate = 0.001


'''

def load_array(fname):
    import bcolz
    return bcolz.open(fname)[:]

# Download my own MNIST
# MNIST Dataset 
PATH = '/Users/krishna/MOOC/DL/Torch/Torch/Data/mnist/' 

TRAIN = PATH + 'train/500/'
TEST = PATH + 'test/500/'
VAL = PATH + 'val/500/'


x_train = load_array(TRAIN + 'x_train.bc/')
y_train = load_array(TRAIN + 'y_train.bc/')
x_test = load_array(TEST + 'x_test.bc/')
y_test = load_array(TEST + 'y_test.bc/')
x_val = load_array(VAL + 'x_val.bc/')
y_val = load_array(VAL + 'y_val.bc/')


# import pdb; pdb.set_trace()
X = torch.from_numpy(x_train)
y = torch.from_numpy(y_train)

train = [X, y]
'''

train = dsets.MNIST(root='../Data', 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train, 
                                           batch_size=batch_size, 
                                           shuffle=True)

class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
        self.out = nn.Softmax()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.out(out)
        return out

net = Net(input_size, hidden_size, num_classes)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step() 

# Yhat
# Acc

