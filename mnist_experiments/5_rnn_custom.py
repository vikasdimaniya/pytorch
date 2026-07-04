import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

input_size = 28
sequence_length = 28
hidden_size = 128
num_classes = 10
num_epochs = 15
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root="../data", train=True, transform=transforms.ToTensor(), download=False)
test_dataset = torchvision.datasets.MNIST(root="../data", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class RNNCustom(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNCustom, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.i2o = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        hidden = torch.zeros(x.size(0), self.hidden_size).to(device)
        for t in range(x.size(1)):
            combined = torch.cat((x[:, t, :], hidden), 1)
            hidden = self.relu(self.i2h(combined))
        output = self.i2o(hidden)
        return output


model = RNNCustom(input_size, hidden_size, num_classes).to(device)
n_params = sum(p.numel() for p in model.parameters())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}')

train_time = time.time() - start_time

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    acc = 100.0 * n_correct / n_samples

print(f'RESULT|RNN_Custom|{n_params}|{acc:.2f}|{train_time:.2f}')
