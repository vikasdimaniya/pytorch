import torch
import torch.nn as nn
import time
from dataset import get_loaders, evaluate_flat, N_BITS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 128
num_epochs = 30
learning_rate = 0.001

train_loader, test_loader = get_loaders()


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(16, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, N_BITS),
        )

    def forward(self, x):
        return self.net(x)


model = MLP().to(device)
n_params = sum(p.numel() for p in model.parameters())

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for x_flat, x_seq, y in train_loader:
        x_flat, y = x_flat.to(device), y.to(device)
        outputs = model(x_flat)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        elapsed = time.time() - start_time
        bit_acc, exact_acc = evaluate_flat(model, test_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} [{elapsed:.0f}s], bit={bit_acc:.2f}%, exact={exact_acc:.2f}%", flush=True)

train_time = time.time() - start_time
bit_acc, exact_acc = evaluate_flat(model, test_loader, device)
print(f"RESULT|MLP|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.2f}")
