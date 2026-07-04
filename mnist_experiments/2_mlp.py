import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

input_size = 28 * 28
hidden_size = 100
num_classes = 10
num_epochs = 3
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root="../data", train=True, transform=transforms.ToTensor(), download=False)
test_dataset = torchvision.datasets.MNIST(root="../data", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size * 4)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu4 = nn.ReLU()
        self.l5 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        out = self.relu1(self.l1(x))
        out = self.relu2(self.l2(out))
        out = self.relu3(self.l3(out))
        out = self.relu4(self.l4(out))
        out = self.l5(out)
        return out


model = MLP(input_size, hidden_size, num_classes).to(device)
n_params = sum(p.numel() for p in model.parameters())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
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
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    acc = 100.0 * n_correct / n_samples

print(f'RESULT|MLP|{n_params}|{acc:.2f}|{train_time:.2f}')
