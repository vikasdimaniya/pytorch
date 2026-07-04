import torch
import torch.nn as nn
import time
from dataset import get_loaders, N_BITS, OUT_BITS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 16
num_epochs = 50
learning_rate = 0.001

train_loader, test_loader = get_loaders(batch_size=256)


def reshape_seq2seq(x):
    bits_a = x[:, :N_BITS]
    bits_b = x[:, N_BITS:]
    paired = torch.stack([bits_a, bits_b], dim=2)
    pad = torch.zeros(x.size(0), 1, 2).to(x.device)
    return torch.cat([paired, pad], dim=1)


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
        out = self.fc(out).squeeze(2)
        return out


model = LSTMSeq2Seq().to(device)
n_params = sum(p.numel() for p in model.parameters())

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = reshape_seq2seq(x)
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        model.eval()
        elapsed = time.time() - start_time
        total_bits = 0; correct_bits = 0; total_samples = 0; exact = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                x = reshape_seq2seq(x)
                preds = (torch.sigmoid(model(x)) > 0.5).float()
                correct_bits += (preds == y).sum().item()
                total_bits += y.numel()
                exact += (preds == y).all(dim=1).sum().item()
                total_samples += y.size(0)
        bit_acc = 100.0 * correct_bits / total_bits
        exact_acc = 100.0 * exact / total_samples
        print(f"Epoch {epoch+1}/{num_epochs} [{elapsed:.0f}s], bit={bit_acc:.2f}%, exact={exact_acc:.2f}%")

train_time = time.time() - start_time
print(f"RESULT|LSTM_S2S|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.2f}")
