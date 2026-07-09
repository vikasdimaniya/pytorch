import torch
import torch.nn as nn
import time
from dataset import get_loaders, evaluate, N_BITS, OUT_BITS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 256
num_epochs = 100
learning_rate = 0.001

train_loader, test_loader = get_loaders(batch_size=256, n_train=80000, n_test=20000)


def reshape_for_seq2seq(x):
    bits_a = x[:, :N_BITS].flip(1)
    bits_b = x[:, N_BITS:].flip(1)
    return torch.cat([bits_a.unsqueeze(2), bits_b.unsqueeze(2)], dim=2)


class LSTMSeq2Seq(nn.Module):
    def __init__(self):
        super(LSTMSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        bits = self.fc(out).squeeze(2)
        return bits


model = LSTMSeq2Seq().to(device)
n_params = sum(p.numel() for p in model.parameters())

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_evaluate(model, test_loader):
    model.eval()
    total_bits = 0
    correct_bits = 0
    total_samples = 0
    exact_matches = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x_seq = reshape_for_seq2seq(x)
            y_msb = y.flip(1)
            outputs = model(x_seq)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_bits += (preds == y_msb).sum().item()
            total_bits += y_msb.numel()
            exact_matches += (preds == y_msb).all(dim=1).sum().item()
            total_samples += y.size(0)
    return 100.0 * correct_bits / total_bits, 100.0 * exact_matches / total_samples


start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x_seq = reshape_for_seq2seq(x)
        y_msb = y.flip(1)
        outputs = model(x_seq)
        loss = criterion(outputs, y_msb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        elapsed = time.time() - start_time
        bit_acc, exact_acc = train_evaluate(model, test_loader)
        print(f"Epoch {epoch+1}/{num_epochs} [{elapsed:.0f}s], bit={bit_acc:.2f}%, exact={exact_acc:.2f}%", flush=True)

train_time = time.time() - start_time
bit_acc, exact_acc = train_evaluate(model, test_loader)
print(f"RESULT|LSTM-S2S|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.2f}")
