"""
32-bit division with per-stage LR warm restarts.
When we switch to a new curriculum stage, the LR resets to max.
Each stage gets its own cosine decay from high to low.
"""
import torch
import torch.nn as nn
import time
import random
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_BITS = 32
hidden_size = 128
learning_rate = 0.003
BATCH_SIZE = 256

CURRICULUM = [
    (2,   30,  5000),
    (4,   40,  10000),
    (8,   50,  20000),
    (16,  60,  30000),
    (32, 200,  60000),   # most epochs here with fresh high LR
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
        x_seq = [[bits_a[t]] + bits_b for t in range(N_BITS)]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(bits_q, dtype=torch.float32)


def sample_pairs(max_bits, n_samples, seed):
    max_val = (1 << max_bits) - 1
    rng = random.Random(seed)
    if max_bits <= 8:
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


def make_loader(max_bits, n_samples):
    pairs = sample_pairs(max_bits, n_samples, seed=42)
    print(f"  Stage: {max_bits}-bit (1-{(1 << max_bits) - 1}), {len(pairs)} samples")
    return DataLoader(DivisionDataset(pairs), batch_size=BATCH_SIZE, shuffle=True)


def make_mixed_test_loader():
    all_pairs = []
    for max_bits, seed_off in [(4, 0), (8, 1), (16, 2), (24, 3), (32, 4)]:
        if max_bits <= 4:
            pairs = sample_pairs(max_bits, 4000, seed=7770 + seed_off)
        else:
            min_val = (1 << (max_bits - 4)) + 1
            max_val = (1 << max_bits) - 1
            rng = random.Random(7770 + max_bits)
            pairs = set()
            while len(pairs) < 4000:
                b = rng.randint(min_val, max_val)
                a = rng.randint(b + 1, max(b + 2, max_val))
                if a <= max_val:
                    pairs.add((a, b))
            pairs = list(pairs)
        all_pairs.extend(pairs)
    random.Random(999).shuffle(all_pairs)
    print(f"Mixed test set: {len(all_pairs)} pairs\n")
    return DataLoader(DivisionDataset(all_pairs), batch_size=BATCH_SIZE, shuffle=False)


class GRUSeq2Seq(nn.Module):
    def __init__(self):
        super(GRUSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(33, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
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


model = GRUSeq2Seq().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params} params, hidden={hidden_size}")

criterion = nn.BCEWithLogitsLoss()
test_loader = make_mixed_test_loader()

start_time = time.time()
best_exact = 0
global_epoch = 0

for stage_bits, stage_epochs, n_samples in CURRICULUM:
    print(f"\n{'='*60}")
    print(f"STAGE: {stage_bits}-bit, {stage_epochs} epochs, LR reset to {learning_rate}")
    print(f"{'='*60}")
    train_loader = make_loader(stage_bits, n_samples)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=stage_epochs, eta_min=1e-5
    )

    for ep in range(1, stage_epochs + 1):
        global_epoch += 1
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if ep % 10 == 0:
            elapsed = time.time() - start_time
            bit_acc, exact_acc = evaluate(model, test_loader)
            best_exact = max(best_exact, exact_acc)
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  [{stage_bits}b] ep {ep}/{stage_epochs} (global {global_epoch}) [{elapsed:.0f}s] lr={lr_now:.5f}, bit={bit_acc:.2f}%, exact={exact_acc:.2f}% (best={best_exact:.2f}%)", flush=True)

train_time = time.time() - start_time
bit_acc, exact_acc = evaluate(model, test_loader)
best_exact = max(best_exact, exact_acc)
print(f"\nFinal: bit={bit_acc:.2f}%, exact={exact_acc:.2f}%, best={best_exact:.2f}%")
print(f"RESULT|GRU-S2S-CurrV3|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.2f}")
