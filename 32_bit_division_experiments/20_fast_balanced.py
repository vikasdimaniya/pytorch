"""
Fast balanced experiment: sample EQUAL numbers across quotient bit-lengths.
The old dataset was ~67% Q=1 (trivial A≈B). This one forces the model
to see hard cases. Target: <10 min total.

Usage: python3 20_fast_balanced.py [hidden_size]
  default hidden_size=256
"""
import torch
import torch.nn as nn
import time
import random
import sys
import os
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(8)
device = torch.device("cpu")

N_BITS = 32
hidden_size = int(sys.argv[1]) if len(sys.argv) > 1 else 256
BATCH_SIZE = 512
TAG = f"h{hidden_size}"

def int_to_bits_msb(n, n_bits):
    return [(n >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]

def bits_to_int_msb(bits):
    n = 0
    for b in bits:
        n = (n << 1) | int(b)
    return n


def sample_balanced(n_per_qlen, max_q_bits, seed):
    """Sample equal counts per quotient bit-length (1..max_q_bits)."""
    rng = random.Random(seed)
    max_val = (1 << N_BITS) - 1
    pairs = []
    for q_bits in range(1, max_q_bits + 1):
        q_lo = 1 << (q_bits - 1)
        q_hi = (1 << q_bits) - 1
        count = 0
        attempts = 0
        while count < n_per_qlen and attempts < n_per_qlen * 100:
            attempts += 1
            q = rng.randint(q_lo, q_hi)
            max_b = max_val // q
            if max_b < 1:
                continue
            b = rng.randint(1, max_b)
            remainder = rng.randint(0, b - 1)
            a = q * b + remainder
            if a > max_val or a <= b:
                continue
            pairs.append((a, b))
            count += 1
    rng.shuffle(pairs)
    return pairs


def sample_curriculum(max_bits, n_samples, seed):
    """For curriculum stages: random pairs within max_bits."""
    max_val = (1 << max_bits) - 1
    rng = random.Random(seed)
    if max_bits <= 8:
        pairs = [(a, b) for b in range(1, max_val + 1) for a in range(b + 1, max_val + 1)]
        rng.shuffle(pairs)
        return pairs[:n_samples]
    pairs = set()
    while len(pairs) < n_samples:
        b = rng.randint(1, max_val)
        a = rng.randint(b + 1, max(b + 2, max_val))
        if a <= max_val:
            pairs.add((a, b))
    return list(pairs)


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
        x_seq = [[bits_a[t]] + bits_b for t in range(N_BITS)]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(bits_q, dtype=torch.float32)


class GRUSeq2Seq(nn.Module):
    def __init__(self):
        super(GRUSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(33, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out).squeeze(2)


def evaluate(model, loader):
    model.eval()
    total_bits = correct_bits = total = exact = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = (torch.sigmoid(model(x)) > 0.5).float()
            correct_bits += (preds == y).sum().item()
            total_bits += y.numel()
            exact += (preds == y).all(dim=1).sum().item()
            total += y.size(0)
    return 100.0 * correct_bits / total_bits, 100.0 * exact / total


def evaluate_by_qlen(model, loader):
    """Per-quotient-bit-length accuracy."""
    model.eval()
    buckets = defaultdict(lambda: {'total': 0, 'exact': 0})
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = (torch.sigmoid(model(x)) > 0.5).float()
            for j in range(y.size(0)):
                q_val = bits_to_int_msb(y[j].cpu().numpy().astype(int).tolist())
                q_bits = q_val.bit_length() if q_val > 0 else 0
                buckets[q_bits]['total'] += 1
                if (preds[j] == y[j]).all():
                    buckets[q_bits]['exact'] += 1
    return buckets


# ============================================================
# Data
# ============================================================
print(f"=== Fast Balanced Experiment, hidden={hidden_size} ===\n")

# Balanced training data: 2500 per quotient bit-length, 1-10 bits
train_pairs = sample_balanced(n_per_qlen=2500, max_q_bits=10, seed=42)
train_loader = DataLoader(DivisionDataset(train_pairs), batch_size=BATCH_SIZE, shuffle=True)
print(f"Train: {len(train_pairs)} samples (balanced across Q bit-lengths)")

# Show distribution
train_dist = defaultdict(int)
for a, b in train_pairs:
    train_dist[(a // b).bit_length()] += 1
for k in sorted(train_dist.keys()):
    print(f"  Q={k}-bit: {train_dist[k]} samples")

# Balanced test data (different seed)
test_pairs = sample_balanced(n_per_qlen=500, max_q_bits=10, seed=9999)
test_loader = DataLoader(DivisionDataset(test_pairs), batch_size=BATCH_SIZE, shuffle=False)
print(f"Test: {len(test_pairs)} samples\n")


# ============================================================
# Model
# ============================================================
model = GRUSeq2Seq().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params} params")

criterion = nn.BCEWithLogitsLoss()
start_time = time.time()
best_exact = 0
ckpt_path = f"balanced_{TAG}.pt"

# ============================================================
# PHASE 1: Quick curriculum (20 epochs, ~2 min)
# ============================================================
print("\nPHASE 1: Curriculum warm-up")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for stage_bits, stage_epochs, n_samp in [(2, 5, 2000), (4, 5, 3000), (8, 5, 5000), (16, 5, 8000)]:
    loader = DataLoader(
        DivisionDataset(sample_curriculum(stage_bits, n_samp, seed=42)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    for ep in range(1, stage_epochs + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    elapsed = time.time() - start_time
    bit_acc, exact_acc = evaluate(model, test_loader)
    best_exact = max(best_exact, exact_acc)
    print(f"  {stage_bits}-bit done [{elapsed:.0f}s] bit={bit_acc:.2f}% exact={exact_acc:.2f}%", flush=True)


# ============================================================
# PHASE 2: Cyclic LR on balanced data (30 epochs, ~4 min)
# ============================================================
phase2_epochs = 30
print(f"\nPHASE 2: Cyclic LR on balanced data ({phase2_epochs} epochs)")
steps_per_epoch = len(train_loader)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=0.0001, max_lr=0.002,
    step_size_up=steps_per_epoch * 8, step_size_down=steps_per_epoch * 8,
    mode='triangular2', cycle_momentum=False,
)

for ep in range(1, phase2_epochs + 1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    if ep % 5 == 0:
        elapsed = time.time() - start_time
        bit_acc, exact_acc = evaluate(model, test_loader)
        saved = ""
        if exact_acc > best_exact:
            best_exact = exact_acc
            torch.save(model.state_dict(), ckpt_path)
            saved = " *** SAVED ***"
        print(f"  ep {ep}/{phase2_epochs} [{elapsed:.0f}s] bit={bit_acc:.2f}% exact={exact_acc:.2f}%{saved} (best={best_exact:.2f}%)", flush=True)


# ============================================================
# PHASE 3: Cosine fine-tune from best checkpoint (20 epochs, ~3 min)
# ============================================================
phase3_epochs = 20
print(f"\nPHASE 3: Cosine fine-tune ({phase3_epochs} epochs)")
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(f"  Loaded best checkpoint (exact={best_exact:.2f}%)")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase3_epochs, eta_min=1e-6)

for ep in range(1, phase3_epochs + 1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    if ep % 5 == 0:
        elapsed = time.time() - start_time
        bit_acc, exact_acc = evaluate(model, test_loader)
        saved = ""
        if exact_acc > best_exact:
            best_exact = exact_acc
            torch.save(model.state_dict(), ckpt_path)
            saved = " *** SAVED ***"
        print(f"  ep {ep}/{phase3_epochs} [{elapsed:.0f}s] bit={bit_acc:.2f}% exact={exact_acc:.2f}%{saved} (best={best_exact:.2f}%)", flush=True)


# ============================================================
# Final evaluation with per-Q-length breakdown
# ============================================================
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

train_time = time.time() - start_time
bit_acc, exact_acc = evaluate(model, test_loader)
print(f"\nFinal: bit={bit_acc:.2f}%, exact={exact_acc:.2f}% [{train_time:.0f}s]")
print(f"RESULT|GRU-Balanced-{TAG}|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.0f}")

buckets = evaluate_by_qlen(model, test_loader)
print(f"\nPer quotient-bit-length accuracy:")
for qb in sorted(buckets.keys()):
    b = buckets[qb]
    rate = 100 * b['exact'] / b['total'] if b['total'] > 0 else 0
    print(f"  Q={qb:2d}-bit: {rate:5.1f}% ({b['exact']}/{b['total']})")
