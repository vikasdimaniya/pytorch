import torch
import torch.nn as nn
import time
from dataset import get_loaders, evaluate, N_BITS, OUT_BITS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 256
num_epochs = 50
learning_rate = 0.001

train_loader, test_loader = get_loaders(batch_size=256)


def reshape_for_rnn(x):
    op_bits = x[:, :2]
    bits_a = x[:, 2:2+N_BITS]
    bits_b = x[:, 2+N_BITS:]
    op_expanded = op_bits.unsqueeze(1).expand(-1, N_BITS, -1)
    a_expanded = bits_a.unsqueeze(2)
    b_expanded = bits_b.unsqueeze(2)
    return torch.cat([op_expanded, a_expanded, b_expanded], dim=2)


class GRUNet(nn.Module):
    def __init__(self):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(4, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, OUT_BITS)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])


model = GRUNet().to(device)
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
        print(f"Epoch {epoch+1}/{num_epochs} [{elapsed:.0f}s], bit={bit_acc:.2f}%, exact={exact_acc:.2f}%", flush=True)

train_time = time.time() - start_time
bit_acc, exact_acc = evaluate(model, test_loader, device, reshape_fn=reshape_for_rnn)
print(f"RESULT|GRU|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.2f}")
