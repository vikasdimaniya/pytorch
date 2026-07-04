import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dataset import get_loaders, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 128
num_epochs = 50
learning_rate = 0.001

train_loader, test_loader = get_loaders(batch_size=256)


def reshape_for_cnn(x):
    # x: (batch, 16) -> (batch, 2, 8): 2 channels (num1 bits, num2 bits), length 8
    bits_a = x[:, :8]   # first 8 bits
    bits_b = x[:, 8:]   # next 8 bits
    return torch.stack([bits_a, bits_b], dim=1)  # (batch, 2, 8)


class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=3, padding=1)  # (batch, 16, 8)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1) # (batch, 32, 8)
        self.fc1 = nn.Linear(32 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 9)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


model = CNN1D().to(device)
n_params = sum(p.numel() for p in model.parameters())

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = reshape_for_cnn(x)
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        bit_acc, exact_acc = evaluate(model, test_loader, device, reshape_fn=reshape_for_cnn)
        print(f"Epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}, bit={bit_acc:.2f}%, exact={exact_acc:.2f}%")

train_time = time.time() - start_time
bit_acc, exact_acc = evaluate(model, test_loader, device, reshape_fn=reshape_for_cnn)
print(f"RESULT|CNN_1D|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.2f}")
