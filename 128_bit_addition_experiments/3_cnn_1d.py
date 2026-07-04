import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dataset import get_loaders, evaluate, N_BITS, OUT_BITS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 256
num_epochs = 50
learning_rate = 0.001

train_loader, test_loader = get_loaders(batch_size=256)


def reshape_for_cnn(x):
    # x: (batch, 256) -> (batch, 2, 128): 2 channels, length 128
    bits_a = x[:, :N_BITS]
    bits_b = x[:, N_BITS:]
    return torch.stack([bits_a, bits_b], dim=1)


class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)   # (batch, 32, 128)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)  # (batch, 64, 128)
        self.pool = nn.AdaptiveAvgPool1d(16)                       # (batch, 64, 16)
        self.fc1 = nn.Linear(64 * 16, hidden_size)
        self.fc2 = nn.Linear(hidden_size, OUT_BITS)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 16)
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

    if (epoch + 1) % 5 == 0:
        elapsed = time.time() - start_time
        bit_acc, exact_acc = evaluate(model, test_loader, device, reshape_fn=reshape_for_cnn)
        print(f"Epoch {epoch+1}/{num_epochs} [{elapsed:.0f}s], loss={loss.item():.4f}, bit={bit_acc:.2f}%, exact={exact_acc:.2f}%")

train_time = time.time() - start_time
bit_acc, exact_acc = evaluate(model, test_loader, device, reshape_fn=reshape_for_cnn)
print(f"RESULT|CNN_1D|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.2f}")
