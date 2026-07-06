"""
GRU seq2seq with curriculum learning:
  Start training on small numbers, gradually increase to full 8-bit range.
  All numbers are still represented as 8-bit MSB-first (with leading zeros).
"""
import torch
import torch.nn as nn
import time
import random
from torch.utils.data import Dataset, DataLoader
from dataset import N_BITS, int_to_bits_msb, evaluate_seq2seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 128
num_epochs = 300
learning_rate = 0.001

CURRICULUM = [
    (1,   50,  4),   # epochs 1-50:   numbers up to 4 bits (1-15)
    (51,  100, 5),   # epochs 51-100: numbers up to 5 bits (1-31)
    (101, 150, 6),   # epochs 101-150: numbers up to 6 bits (1-63)
    (151, 200, 7),   # epochs 151-200: numbers up to 7 bits (1-127)
    (201, 300, 8),   # epochs 201-300: numbers up to 8 bits (1-255)
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
    pairs = []
    for b in range(1, max_val + 1):
        for a in range(b + 1, max_val + 1):
            pairs.append((a, b))
    random.seed(42)
    random.shuffle(pairs)
    n = len(pairs)
    print(f"  Curriculum stage: {max_bits}-bit numbers (1-{max_val}), {n} pairs")
    return DataLoader(DivisionDataset(pairs), batch_size=batch_size, shuffle=True)


# Fixed test set: always full 8-bit range
def make_test_loader(batch_size=128):
    pairs = []
    for b in range(1, 256):
        for a in range(b + 1, 256):
            pairs.append((a, b))
    random.seed(123)
    random.shuffle(pairs)
    n = len(pairs)
    split = int(0.8 * n)
    test_pairs = pairs[split:]
    return DataLoader(DivisionDataset(test_pairs), batch_size=batch_size, shuffle=False)


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

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

test_loader = make_test_loader()

start_time = time.time()
best_exact = 0
current_stage = -1
train_loader = None

for epoch in range(1, num_epochs + 1):
    # Check if we need to switch curriculum stage
    for stage_start, stage_end, max_bits in CURRICULUM:
        if stage_start <= epoch <= stage_end:
            stage_idx = CURRICULUM.index((stage_start, stage_end, max_bits))
            if stage_idx != current_stage:
                current_stage = stage_idx
                print(f"\n--- Epoch {epoch}: switching to {max_bits}-bit curriculum ---")
                train_loader = make_loader(max_bits)
            break

    model.train()
    for x_flat, x_seq, y in train_loader:
        x_seq, y = x_seq.to(device), y.to(device)
        outputs = model(x_seq)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        elapsed = time.time() - start_time
        bit_acc, exact_acc = evaluate_seq2seq(model, test_loader, device)
        best_exact = max(best_exact, exact_acc)
        print(f"Epoch {epoch}/{num_epochs} [{elapsed:.0f}s], bit={bit_acc:.2f}%, exact={exact_acc:.2f}% (best={best_exact:.2f}%)", flush=True)

train_time = time.time() - start_time
bit_acc, exact_acc = evaluate_seq2seq(model, test_loader, device)
print(f"\nRESULT|GRU-S2S-Curriculum|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.2f}")
