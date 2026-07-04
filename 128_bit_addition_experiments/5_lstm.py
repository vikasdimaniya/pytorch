import torch
import torch.nn as nn
import time
from dataset import get_loaders, evaluate, N_BITS, OUT_BITS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 128
num_epochs = 50
learning_rate = 0.001

train_loader, test_loader = get_loaders(batch_size=256)


def reshape_for_rnn(x):
    bits_a = x[:, :N_BITS]
    bits_b = x[:, N_BITS:]
    return torch.stack([bits_a, bits_b], dim=2)


class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, OUT_BITS)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        return self.fc(out)


model = LSTMNet().to(device)
n_params = sum(p.numel() for p in model.parameters())

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = reshape_for_rnn(x)
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        elapsed = time.time() - start_time
        bit_acc, exact_acc = evaluate(model, test_loader, device, reshape_fn=reshape_for_rnn)
        print(f"Epoch {epoch+1}/{num_epochs} [{elapsed:.0f}s], loss={loss.item():.4f}, bit={bit_acc:.2f}%, exact={exact_acc:.2f}%")

train_time = time.time() - start_time
bit_acc, exact_acc = evaluate(model, test_loader, device, reshape_fn=reshape_for_rnn)
print(f"RESULT|LSTM|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.2f}")
