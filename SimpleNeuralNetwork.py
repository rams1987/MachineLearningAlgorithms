# Code a NN model using Pytorch
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
        x = torch.relu(self.fc1(x))
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

#print(train_loader)

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
        print(data.shape)
        #reshape the data. Single vector
        data = data.reshape(data.shape[0],-1)
        #print(data.shape)

        #Forward pass
        scores = model(data)
        loss = criterion(scores, targets) #calculate loss using scores from the model vs targets

        # Backward pass
        optimizer.zero_grad() # Set all the grdient to zero for each batch.
        loss.backward() # Backpropagation algorithm

        # Gradient descent
        optimizer.step()


# Model parameters
def model_parameters(model):
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict() [param_tensor].size())

model_parameters(model)


# Save a model
def save_model(model, path):
    torch.save(model.state_dict(),path)

model_path = '/Users/rams/Documents/Projects/MachineLearning/Models/model'
save_model(model,model_path)

# Check accuracy of the model

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval() # switch model to evaluation mode. Remove unwanted layers.

    with torch.no_grad(): #switch off gradient calculation
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
