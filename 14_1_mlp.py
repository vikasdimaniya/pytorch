## 4 layers
# accuracy = 39% with 0.001 learning rate with Adam, batch size 4
# accuracy = 18%, learning rate 0.005 with adam, batch size 4
# accuracy = 46.75% learning rate 0.001 with batch size 32
# accuracy = 46% learning rate 0.001 with batch size 64

## 5 layers
# accuracy = 44% learning rate 0.001 with batch size 64

## 6 layers
# accuracy = 44% learning rate 0.001 with batch size 64
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()
#device
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
#hyperparameters
input_size = 3*32*32
hidden_size = 100
num_classes = 10
num_epochs = 4
batch_size = 128
learning_rate = 0.0005

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

print(samples.shape, labels.shape)


# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(samples[i][0],cmap='gray')
# plt.show()
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        prev_size = int(input_size/2)
        print ("hidden",prev_size)
        self.l1 = nn.Linear(input_size, 2500)
        
        prev_size = int(prev_size/2)
        print (prev_size)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(2500, 2000)
        
        prev_size = int(prev_size/2)
        print (prev_size)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(2000, prev_size)
        
        prev_size = int(prev_size/2)
        print (prev_size)
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(prev_size*2, prev_size)
        
        prev_size = int(prev_size/2)
        print (prev_size)
        self.relu4 = nn.ReLU()
        self.l5 = nn.Linear(prev_size*2, prev_size)

        prev_size = int(prev_size/2)
        print (prev_size)
        self.relu5 = nn.ReLU()
        self.l6 = nn.Linear(prev_size*2, num_classes)
    def forward(self,x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        out = self.relu3(out)
        out = self.l4(out)
        out = self.relu4(out)
        out = self.l5(out)
        out = self.relu5(out)
        out = self.l6(out)
        return out

class ConvNet(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass

model = NeuralNet(input_size, num_classes).to(device)
#model = ConvNet(input_size, hidden_size, num_classes).to(device)

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
        images = images.reshape(-1, 3*32*32).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if ((i+1)%500==0):
            print(f'This is epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

        
# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.reshape(-1, 3*32*32).to(device)
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