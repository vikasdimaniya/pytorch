"""
GRU seq2seq with hidden_size=32 (~4,100 params).
Still 6x fewer params than training samples — can't memorize.
Curriculum + cosine annealing LR + 500 epochs.
"""
import torch
import torch.nn as nn
import math
import time
import random
from torch.utils.data import Dataset, DataLoader
from dataset import N_BITS, int_to_bits_msb, evaluate_seq2seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 32
num_epochs = 500
learning_rate = 0.003

CURRICULUM = [
    (1,   50,  4),
    (51,  100, 5),
    (101, 150, 6),
    (151, 220, 7),
    (221, 500, 8),
]


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
        x_flat = torch.tensor(bits_a + bits_b, dtype=torch.float32)
        x_seq = torch.tensor(x_seq, dtype=torch.float32)
        y = torch.tensor(bits_q, dtype=torch.float32)
        return x_flat, x_seq, y


def make_loader(max_bits, batch_size=128):
    max_val = (1 << max_bits) - 1
    pairs = [(a, b) for b in range(1, max_val + 1) for a in range(b + 1, max_val + 1)]
    random.seed(42)
    random.shuffle(pairs)
    print(f"  Stage: {max_bits}-bit (1-{max_val}), {len(pairs)} pairs")
    return DataLoader(DivisionDataset(pairs), batch_size=batch_size, shuffle=True)


def make_test_loader(batch_size=128):
    pairs = [(a, b) for b in range(1, 256) for a in range(b + 1, 256)]
    random.seed(123)
    random.shuffle(pairs)
    split = int(0.8 * len(pairs))
    return DataLoader(DivisionDataset(pairs[split:]), batch_size=batch_size, shuffle=False)


class GRUSeq2Seq(nn.Module):
    def __init__(self):
        super(GRUSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(9, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        bits = self.fc(out).squeeze(2)
        return bits


model = GRUSeq2Seq().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params} params, {n_params/25908:.4f} params/sample\n")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

test_loader = make_test_loader()

start_time = time.time()
best_exact = 0
current_stage = -1
train_loader = None

for epoch in range(1, num_epochs + 1):
    for i, (s, e, bits) in enumerate(CURRICULUM):
        if s <= epoch <= e and i != current_stage:
            current_stage = i
            print(f"\n--- Epoch {epoch}: switching to {bits}-bit ---")
            train_loader = make_loader(bits)
            break

    model.train()
    for x_flat, x_seq, y in train_loader:
        x_seq, y = x_seq.to(device), y.to(device)
        outputs = model(x_seq)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    if epoch % 20 == 0:
        elapsed = time.time() - start_time
        bit_acc, exact_acc = evaluate_seq2seq(model, test_loader, device)
        best_exact = max(best_exact, exact_acc)
        print(f"Epoch {epoch}/{num_epochs} [{elapsed:.0f}s], bit={bit_acc:.2f}%, exact={exact_acc:.2f}% (best={best_exact:.2f}%)", flush=True)

train_time = time.time() - start_time
bit_acc, exact_acc = evaluate_seq2seq(model, test_loader, device)
print(f"\nRESULT|GRU-S2S-h32|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.2f}")
