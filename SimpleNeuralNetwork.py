# Code a CNN model using Pytorch
# https://www.youtube.com/watch?v=Jy4wM2X21u0&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transform

#Create a fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self,x):
        x = torch.nn.ReLU(self.fc1(x))
        x = self.fc2(x)
        return x

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size = 784 #MNIST data size 28x28 = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

#load data. # Convert data to pytorch tensor
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transform.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transform.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

print(train_loader)

#Initiaze network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Train Network
# data - is the images
# targets - ground truth labels
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        #reshape the data. Single vector
        data = data.reshape(data.shape[0],-1)

        #Forward pass
        scores = model(data)
        loss = criterion(scores, targets) #calculate loss using scores from the model vs targets

        # Backward pass
        optimizer.zero_grad() # Set all the grdient to zero for each batch.
        loss.backward() # Backpropagation algorithm

        # Gradient descent
        optimizer.step()

# Check accuracy of the model
"""
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
"""