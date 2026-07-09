"""
32-bit division: applying all learnings from 8-bit experiments.
- Seq2seq: full divisor at every step + 1 dividend bit per step
- MSB-first processing (natural for long division)
- Curriculum learning: ramp from 8-bit to 32-bit
- Cosine annealing LR
- hidden_size=64
- A > B always (quotient >= 1)
"""
import torch
import torch.nn as nn
import time
import random
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_BITS = 32
hidden_size = 64
num_epochs = 200
learning_rate = 0.003
BATCH_SIZE = 256

CURRICULUM = [
    (1,   20,   8, 10000),
    (21,  40,  12, 20000),
    (41,  60,  16, 30000),
    (61,  80,  20, 40000),
    (81,  100, 24, 50000),
    (101, 130, 28, 60000),
    (131, 200, 32, 80000),
]


def int_to_bits_msb(n, n_bits):
    return [(n >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]


class DivisionDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        q = a // b

        bits_a = int_to_bits_msb(a, N_BITS)
        bits_b = int_to_bits_msb(b, N_BITS)
        bits_q = int_to_bits_msb(q, N_BITS)

        x_seq = []
        for t in range(N_BITS):
            x_seq.append([bits_a[t]] + bits_b)

        x_seq = torch.tensor(x_seq, dtype=torch.float32)   # (32, 33)
        y = torch.tensor(bits_q, dtype=torch.float32)       # (32,)
        return x_seq, y


def make_loader(max_bits, n_samples, batch_size=BATCH_SIZE):
    """Sample random pairs where A > B, both in [1, 2^max_bits - 1]."""
    max_val = (1 << max_bits) - 1
    pairs = set()
    random.seed(42)
    while len(pairs) < n_samples:
        b = random.randint(1, max_val)
        a = random.randint(b + 1, max(b + 2, max_val))
        if a <= max_val:
            pairs.add((a, b))
    pairs = list(pairs)
    random.shuffle(pairs)
    print(f"  Stage: {max_bits}-bit (1-{max_val}), {len(pairs)} samples")
    return DataLoader(DivisionDataset(pairs), batch_size=batch_size, shuffle=True)


def make_test_loader(n_samples=20000, batch_size=BATCH_SIZE):
    """Fixed test set: full 32-bit range, A > B."""
    max_val = (1 << N_BITS) - 1
    pairs = set()
    random.seed(999)
    while len(pairs) < n_samples:
        b = random.randint(1, max_val)
        a = random.randint(b + 1, max(b + 2, max_val))
        if a <= max_val:
            pairs.add((a, b))
    pairs = list(pairs)
    return DataLoader(DivisionDataset(pairs), batch_size=batch_size, shuffle=False)


def evaluate(model, test_loader):
    model.eval()
    total_bits = 0
    correct_bits = 0
    total_samples = 0
    exact_matches = 0
    with torch.no_grad():
        for x_seq, y in test_loader:
            x_seq, y = x_seq.to(device), y.to(device)
            outputs = model(x_seq)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_bits += (preds == y).sum().item()
            total_bits += y.numel()
            exact_matches += (preds == y).all(dim=1).sum().item()
            total_samples += y.size(0)
    return 100.0 * correct_bits / total_bits, 100.0 * exact_matches / total_samples


class GRUSeq2Seq(nn.Module):
    def __init__(self):
        super(GRUSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(33, hidden_size, batch_first=True)  # 1 + 32 = 33 inputs
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        bits = self.fc(out).squeeze(2)
        return bits


model = GRUSeq2Seq().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params} params, hidden_size={hidden_size}")
print(f"Test set: 20,000 full 32-bit pairs (A > B)\n")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

test_loader = make_test_loader()

start_time = time.time()
best_exact = 0
current_stage = -1
train_loader = None

for epoch in range(1, num_epochs + 1):
    for i, (s, e, bits, n_samp) in enumerate(CURRICULUM):
        if s <= epoch <= e and i != current_stage:
            current_stage = i
            print(f"\n--- Epoch {epoch}: switching to {bits}-bit curriculum ---")
            train_loader = make_loader(bits, n_samp)
            break

    model.train()
    for x_seq, y in train_loader:
        x_seq, y = x_seq.to(device), y.to(device)
        outputs = model(x_seq)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    if epoch % 10 == 0:
        elapsed = time.time() - start_time
        bit_acc, exact_acc = evaluate(model, test_loader)
        best_exact = max(best_exact, exact_acc)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{num_epochs} [{elapsed:.0f}s] lr={lr_now:.5f}, bit={bit_acc:.2f}%, exact={exact_acc:.2f}% (best={best_exact:.2f}%)", flush=True)

train_time = time.time() - start_time
bit_acc, exact_acc = evaluate(model, test_loader)
best_exact = max(best_exact, exact_acc)
print(f"\nFinal: bit={bit_acc:.2f}%, exact={exact_acc:.2f}%, best={best_exact:.2f}%")
print(f"RESULT|GRU-S2S-Curriculum-32bit|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.2f}")
