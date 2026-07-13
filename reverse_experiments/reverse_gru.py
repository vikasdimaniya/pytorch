"""
GRU String Reversal — Bottleneck Experiment.
Input: sequence of random digits 0-9, one per timestep.
Output: all digits in reverse order, produced from the final hidden state.

Usage: python3 reverse_gru.py [hidden_size] [seq_len]
  defaults: hidden_size=32, seq_len=32
"""
import torch
import torch.nn as nn
import time
import random
import sys
import json
from torch.utils.data import Dataset, DataLoader

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    torch.set_num_threads(4)
    device = torch.device("cpu")

hidden_size = int(sys.argv[1]) if len(sys.argv) > 1 else 32
seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 32
BATCH_SIZE = 256
N_TRAIN = 50000
N_TEST = 10000
EPOCHS = 100
NUM_DIGITS = 10


class ReverseDataset(Dataset):
    def __init__(self, n_samples, seq_len, seed):
        rng = random.Random(seed)
        self.data = []
        for _ in range(n_samples):
            seq = [rng.randint(0, NUM_DIGITS - 1) for _ in range(seq_len)]
            self.data.append(seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        x = torch.tensor(seq, dtype=torch.long)
        y = torch.tensor(list(reversed(seq)), dtype=torch.long)
        return x, y


class GRUReverser(nn.Module):
    def __init__(self, hidden_size, seq_len):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.embed = nn.Embedding(NUM_DIGITS, 16)
        self.gru = nn.GRU(16, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, seq_len * NUM_DIGITS)

    def forward(self, x):
        emb = self.embed(x)
        h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        _, h_final = self.gru(emb, h0)
        h_final = h_final.squeeze(0)
        out = self.fc(h_final)
        return out.view(-1, self.seq_len, NUM_DIGITS)


def evaluate(model, loader):
    model.eval()
    total = 0
    exact_correct = 0
    pos_correct = torch.zeros(seq_len)
    pos_total = torch.zeros(seq_len)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=2)
            match = (preds == y)
            exact_correct += match.all(dim=1).sum().item()
            total += y.size(0)
            pos_correct += match.sum(dim=0).cpu().float()
            pos_total += y.size(0)

    exact_acc = 100.0 * exact_correct / total
    pos_acc = 100.0 * pos_correct / pos_total
    return exact_acc, pos_acc


# Data
train_ds = ReverseDataset(N_TRAIN, seq_len, seed=42)
test_ds = ReverseDataset(N_TEST, seq_len, seed=9999)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = GRUReverser(hidden_size, seq_len).to(device)
n_params = sum(p.numel() for p in model.parameters())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"h={hidden_size}, seq={seq_len}, params={n_params}, device={device}", flush=True)

start_time = time.time()
best_exact = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, NUM_DIGITS), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0 or epoch == EPOCHS:
        exact_acc, pos_acc = evaluate(model, test_loader)
        best_exact = max(best_exact, exact_acc)
        elapsed = time.time() - start_time
        print(f"  ep {epoch}/{EPOCHS} [{elapsed:.0f}s] exact={exact_acc:.2f}% (best={best_exact:.2f}%)", flush=True)

# Final evaluation with per-position breakdown
exact_acc, pos_acc = evaluate(model, test_loader)
best_exact = max(best_exact, exact_acc)
train_time = time.time() - start_time

print(f"\nFinal: exact={exact_acc:.2f}% (best={best_exact:.2f}%) [{train_time:.0f}s]")

print(f"\nPer-position accuracy (output pos 0 = last input, pos {seq_len-1} = first input):")
for i in range(seq_len):
    bar = "#" * int(pos_acc[i] / 2)
    print(f"  pos {i:3d}: {pos_acc[i]:5.1f}% {bar}")

# Output JSON result for grid runner
result = {
    "hidden_size": hidden_size,
    "seq_len": seq_len,
    "params": n_params,
    "best_exact": best_exact,
    "final_exact": exact_acc,
    "time": round(train_time, 1),
    "pos_acc": [round(p, 2) for p in pos_acc.tolist()],
}
print(f"\nRESULT_JSON|{json.dumps(result)}")
