"""
32-bit division with improved curriculum:
- Start from 2-bit, ramp through 4, 8, 16, 32
- More epochs at each lower stage
- Test set is a MIX of all bit sizes (not just 32-bit)
- hidden_size=128, 500 epochs
"""
import torch
import torch.nn as nn
import time
import random
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_BITS = 32
hidden_size = 128
num_epochs = 500
learning_rate = 0.003
BATCH_SIZE = 256

CURRICULUM = [
    (1,   50,   2,  5000),   # 2-bit: numbers 1-3, 50 epochs
    (51,  110,  4,  10000),  # 4-bit: numbers 1-15, 60 epochs
    (111, 180,  8,  20000),  # 8-bit: numbers 1-255, 70 epochs
    (181, 260, 16,  40000),  # 16-bit: numbers 1-65535, 80 epochs
    (261, 500, 32,  80000),  # 32-bit: numbers 1-4B, 240 epochs
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

        x_seq = torch.tensor(x_seq, dtype=torch.float32)
        y = torch.tensor(bits_q, dtype=torch.float32)
        return x_seq, y


def sample_pairs(max_bits, n_samples, seed):
    """Sample random pairs where A > B, both in [1, 2^max_bits - 1]."""
    max_val = (1 << max_bits) - 1
    rng = random.Random(seed)
    if max_bits <= 8:
        # small enough to enumerate all pairs
        pairs = [(a, b) for b in range(1, max_val + 1) for a in range(b + 1, max_val + 1)]
        rng.shuffle(pairs)
        return pairs[:n_samples]
    else:
        pairs = set()
        while len(pairs) < n_samples:
            b = rng.randint(1, max_val)
            a = rng.randint(b + 1, max(b + 2, max_val))
            if a <= max_val:
                pairs.add((a, b))
        return list(pairs)


def make_loader(max_bits, n_samples, batch_size=BATCH_SIZE):
    pairs = sample_pairs(max_bits, n_samples, seed=42)
    print(f"  Stage: {max_bits}-bit (1-{(1 << max_bits) - 1}), {len(pairs)} samples")
    return DataLoader(DivisionDataset(pairs), batch_size=batch_size, shuffle=True)


def make_mixed_test_loader(batch_size=BATCH_SIZE):
    """Test set with equal representation from each bit range."""
    all_pairs = []
    ranges = [
        (2, 4000),    # 2-4 bit numbers (1-15)
        (8, 4000),    # 5-8 bit numbers (16-255)
        (16, 4000),   # 9-16 bit numbers (256-65535)
        (24, 4000),   # 17-24 bit numbers (65536-16M)
        (32, 4000),   # 25-32 bit numbers (16M-4B)
    ]
    for max_bits, n in ranges:
        # for lower ranges, use the full range; for higher ranges, sample from the upper half
        if max_bits <= 4:
            pairs = sample_pairs(max_bits, n, seed=7770)
        else:
            min_val = (1 << (max_bits - 4)) + 1  # ensure numbers actually use these bits
            max_val = (1 << max_bits) - 1
            rng = random.Random(7770 + max_bits)
            pairs = set()
            while len(pairs) < n:
                b = rng.randint(min_val, max_val)
                a = rng.randint(b + 1, max(b + 2, max_val))
                if a <= max_val:
                    pairs.add((a, b))
            pairs = list(pairs)
        all_pairs.extend(pairs)

    random.Random(999).shuffle(all_pairs)
    print(f"Mixed test set: {len(all_pairs)} pairs across all bit ranges\n")
    return DataLoader(DivisionDataset(all_pairs), batch_size=batch_size, shuffle=False)


class GRUSeq2Seq(nn.Module):
    def __init__(self):
        super(GRUSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(33, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        bits = self.fc(out).squeeze(2)
        return bits


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


model = GRUSeq2Seq().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params} params, hidden_size={hidden_size}")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

test_loader = make_mixed_test_loader()

start_time = time.time()
best_exact = 0
current_stage = -1
train_loader = None

for epoch in range(1, num_epochs + 1):
    for i, (s, e, bits, n_samp) in enumerate(CURRICULUM):
        if s <= epoch <= e and i != current_stage:
            current_stage = i
            print(f"\n--- Epoch {epoch}: switching to {bits}-bit ---")
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
print(f"RESULT|GRU-S2S-CurrV2-32bit|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.2f}")
