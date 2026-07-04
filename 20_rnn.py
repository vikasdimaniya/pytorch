import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 28        # one row of pixels
sequence_length = 28   # number of rows (time steps)
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=False)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.i2o = nn.Linear(hidden_size, num_classes)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden_tensor = self.i2h(combined)
        hidden_tensor = self.relu(hidden_tensor)
        output_tensor = self.i2o(hidden_tensor)
        return output_tensor, hidden_tensor

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

model = RNN(input_size, hidden_size, num_classes).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(images, labels):
    # images shape: (batch_size, 28, 28) — 28 rows, each row 28 pixels
    hidden_tensor = model.init_hidden(images.size(0))

    for t in range(images.size(1)):
        output, hidden_tensor = model(images[:, t, :], hidden_tensor)

    loss = criterion(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # (100, 1, 28, 28) -> (100, 28, 28): batch of 28-step sequences
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        output, loss = train(images, labels)

        if (i+1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss:.4f}')

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        hidden_tensor = model.init_hidden(images.size(0))
        for t in range(images.size(1)):
            outputs, hidden_tensor = model(images[:, t, :], hidden_tensor)

        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
