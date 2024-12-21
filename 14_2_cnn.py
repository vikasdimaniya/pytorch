# this one uses 5*5 size filters
# acc with batch size 4096 is 38%
# acc with batch size 4 is 55%
# acc with batch size 1 is 50% with significan increase in time by 28 times when compared to batch of 4
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

start_time = time.time()
#device
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
#hyperparameters

num_epochs = 8
batch_size = 1
learning_rate = 0.001

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform = transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, transform = transforms.ToTensor())

train_loader=torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(train_loader)
examples = iter(train_loader)
print(examples)
samples, labels = next(examples)

print("Printing samples and shape")
print(samples.shape, labels.shape)


# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(samples[i][0],cmap='gray')
# plt.show()

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 4 images of 3*32*32 RGB*W*H
        self.pool = nn.MaxPool2d(2,2) # 4 images of 6*28*28
        self.conv2 = nn.Conv2d(6, 16, 5) # 4 images of 16*10*10
        #there will be one more pooling here that will decrease the size, 16*5*5
        self.l1 = nn.Linear(16*5*5, 120)
        self.l2 = nn.Linear(120, 84)
        self.l3 = nn.Linear(84, 10) # 10 classes

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
print(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)


n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # image is 100, 1, 28, 28 shape
        # we need 784
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if ((i+1)%2000==0):
            print(f'This is epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

        
# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)

        # fetching the class with the max value out of all the classes, as our output has multiple values i.e, num_classes
        _, predicted = torch.max(outputs, 1)
        n_samples +=labels.shape[0]
        #print(predictions==labels)
        n_correct += (predicted == labels).sum().item()
        for i in range(labels.size(0)):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] +=1
            n_class_samples[label] += 1
    acc = 100.0 * n_correct/n_samples
    print(f'accuracy = {acc}')

    for i in range(10):
        acc = 100.0 * n_class_correct[i]/n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")