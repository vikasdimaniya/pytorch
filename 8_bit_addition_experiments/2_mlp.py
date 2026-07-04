import torch
import torch.nn as nn
import time
from dataset import get_loaders, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 128
num_epochs = 50
learning_rate = 0.001

train_loader, test_loader = get_loaders(batch_size=256)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(16, hidden_size)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(hidden_size // 2, 9)

    def forward(self, x):
        out = self.relu1(self.l1(x))
        out = self.relu2(self.l2(out))
        out = self.relu3(self.l3(out))
        return self.l4(out)


model = MLP().to(device)
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

    if (epoch + 1) % 10 == 0:
        bit_acc, exact_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}, bit={bit_acc:.2f}%, exact={exact_acc:.2f}%")

train_time = time.time() - start_time
bit_acc, exact_acc = evaluate(model, test_loader, device)
print(f"RESULT|MLP|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.2f}")
