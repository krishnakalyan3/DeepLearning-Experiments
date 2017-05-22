#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

num_classes = 10

class Mnist(nn.Module):

	def __init__(self):
		super(Mnist, self).__init__()
		self.fc1 = nn.Linear(784, 500)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(500, 10)

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)

model = Mnist()
print(model)

print("######################### Model Parameters ########################")

loss_type = nn.CrossEntropyLoss()
for p in model.parameters():
    print(p.size())

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


train = dsets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train, 
                                           batch_size=128, 
                                           shuffle=True)

# Finally Train Model
for epoch in range(10):
	for i, (data, target) in enumerate(train_loader):
		#print(labels)
		data, target = Variable(data.view(-1, 28*28)), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

