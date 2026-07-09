import torch
import torch.nn as nn
import time
from dataset import get_loaders, evaluate, OUT_BITS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 256
num_epochs = 50
learning_rate = 0.001
INPUT_SIZE = 64  # 32 + 32

train_loader, test_loader = get_loaders(batch_size=256)


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.l1 = nn.Linear(INPUT_SIZE, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, OUT_BITS)

    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))


model = LinearNet().to(device)
n_params = sum(p.numel() for p in model.parameters())

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        elapsed = time.time() - start_time
        bit_acc, exact_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} [{elapsed:.0f}s], bit={bit_acc:.2f}%, exact={exact_acc:.2f}%", flush=True)

train_time = time.time() - start_time
bit_acc, exact_acc = evaluate(model, test_loader, device)
print(f"RESULT|Linear|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.2f}")
