

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import sys

def load_array(fname):
    import bcolz
    return bcolz.open(fname)[:]

# Hyper Parameters 
input_size = 784
hidden_size = 500
num_epochs = 5
learning_rate = 0.001

# MNIST Dataset 
PATH = '/Users/krishna/MOOC/DL/Torch/Data/mnist/'

TRAIN = PATH + 'train/500/'
TEST = PATH + 'test/500/'
VAL = PATH + 'val/500/'

x_train = load_array(TRAIN + 'x_train.bc/')
y_train = load_array(TRAIN + 'y_train.bc/')

examples = x_train.shape[0]
b = np.zeros((examples, 10))
b[np.arange(examples), y_train] = 1
y_train = b
print(y_train.shape)

img_rows, img_cols = 28, 28

#x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).long()

trainset = torch.utils.data.TensorDataset(x_train, y_train)


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                           batch_size=128,
                                           shuffle=True, num_workers=2)

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
net = Net(input_size, hidden_size)

    
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels.long())
        print(labels.size())
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        print(outputs.size())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(trainset)//128, loss.data[0]))

