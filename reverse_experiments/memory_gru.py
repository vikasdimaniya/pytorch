"""
Memory-Augmented GRU for String Reversal.

Architecture:
  - GRU cell processes input one digit at a time
  - External memory bank: seq_len slots, each storing a vector
  - Encoding: GRU reads digit, decides WHERE to write in memory
  - Decoding: GRU decides WHERE to read from memory, predicts digit
  - Addressing is soft (differentiable attention over slots)

Compares against vanilla GRU baseline.

Usage: python3 memory_gru.py [hidden_size] [seq_len]
  defaults: hidden_size=32, seq_len=8
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import sys
from torch.utils.data import Dataset, DataLoader

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    torch.set_num_threads(4)
    device = torch.device("cpu")

hidden_size = int(sys.argv[1]) if len(sys.argv) > 1 else 32
seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BATCH = 256
N_TRAIN = 50000
N_TEST = 10000
EPOCHS = 100
NUM_DIGITS = 10
SLOT_SIZE = 16
GO_TOKEN = NUM_DIGITS


class ReverseDataset(Dataset):
    def __init__(self, n, seq_len, seed):
        rng = random.Random(seed)
        self.data = [[rng.randint(0, 9) for _ in range(seq_len)] for _ in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        s = self.data[i]
        return torch.tensor(s, dtype=torch.long), torch.tensor(s[::-1], dtype=torch.long)


class MemoryGRU(nn.Module):
    """
    GRU + external memory bank.

    Encoding (t = 0..seq_len-1):
      input  = [digit_embedding, read_value_from_prev_step]
      h      = GRUCell(input, h_prev)
      addr   = softmax(linear(h))          -- WHERE in memory
      w_gate = sigmoid(linear(h))          -- HOW MUCH to write (vs keep old)
      w_val  = linear(h)                   -- WHAT to write
      memory updated at addr with w_gate blending

    Decoding (t = 0..seq_len-1):
      input  = [GO_embedding, read_value_from_prev_step]
      h      = GRUCell(input, h_prev)
      addr   = softmax(linear(h))          -- WHERE to read
      read   = weighted sum of memory at addr
      output = linear(read + h) -> digit prediction
    """

    def __init__(self, hidden_size, seq_len):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.embed = nn.Embedding(NUM_DIGITS + 1, 16)
        self.gru = nn.GRUCell(16 + SLOT_SIZE, hidden_size)

        self.addr_head = nn.Linear(hidden_size, seq_len)
        self.write_gate = nn.Linear(hidden_size, 1)
        self.write_head = nn.Linear(hidden_size, SLOT_SIZE)

        self.output_head = nn.Linear(SLOT_SIZE + hidden_size, NUM_DIGITS)

    def forward(self, x, return_diag=False):
        B = x.size(0)
        dev = x.device

        memory = torch.zeros(B, self.seq_len, SLOT_SIZE, device=dev)
        h = torch.zeros(B, self.hidden_size, device=dev)
        read_val = torch.zeros(B, SLOT_SIZE, device=dev)

        enc_addrs, enc_wgates, dec_addrs = [], [], []

        # --- ENCODE: read digits, write to memory ---
        for t in range(self.seq_len):
            emb = self.embed(x[:, t])
            h = self.gru(torch.cat([emb, read_val], 1), h)

            addr = F.softmax(self.addr_head(h), dim=1)
            wg = torch.sigmoid(self.write_gate(h))
            wv = self.write_head(h)

            addr_e = addr.unsqueeze(-1)
            wv_e = wv.unsqueeze(1)
            memory = memory * (1 - wg.unsqueeze(-1) * addr_e) + \
                     wg.unsqueeze(-1) * addr_e * wv_e

            read_val = torch.einsum("bi,bij->bj", addr, memory)

            if return_diag:
                enc_addrs.append(addr[0].detach().cpu())
                enc_wgates.append(wg[0, 0].detach().cpu().item())

        # --- DECODE: read from memory, predict reversed digits ---
        go = torch.full((B,), GO_TOKEN, dtype=torch.long, device=dev)
        outputs = []

        for t in range(self.seq_len):
            emb = self.embed(go)
            h = self.gru(torch.cat([emb, read_val], 1), h)

            addr = F.softmax(self.addr_head(h), dim=1)
            read_val = torch.einsum("bi,bij->bj", addr, memory)

            outputs.append(self.output_head(torch.cat([read_val, h], 1)))

            if return_diag:
                dec_addrs.append(addr[0].detach().cpu())

        logits = torch.stack(outputs, dim=1)

        if return_diag:
            return logits, {
                "enc_addr": enc_addrs, "enc_wgate": enc_wgates,
                "dec_addr": dec_addrs,
            }
        return logits


class VanillaGRU(nn.Module):
    """Baseline: compress everything into final hidden state."""

    def __init__(self, hidden_size, seq_len):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.embed = nn.Embedding(NUM_DIGITS, 16)
        self.gru = nn.GRU(16, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, seq_len * NUM_DIGITS)

    def forward(self, x, return_diag=False):
        emb = self.embed(x)
        h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        _, hf = self.gru(emb, h0)
        logits = self.fc(hf.squeeze(0)).view(-1, self.seq_len, NUM_DIGITS)
        if return_diag:
            return logits, {}
        return logits


def evaluate(model, loader):
    model.eval()
    total = exact = 0
    pos_correct = torch.zeros(seq_len)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=2)
            match = preds == y
            exact += match.all(dim=1).sum().item()
            total += y.size(0)
            pos_correct += match.sum(dim=0).cpu().float()
    return 100.0 * exact / total, 100.0 * pos_correct / total


def train_and_eval(model, name, train_loader, test_loader):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 60}")
    print(f"{name}: h={hidden_size}, seq={seq_len}, params={n_params:,}, device={device}")
    print(f"{'=' * 60}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start = time.time()
    best = 0.0

    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x).reshape(-1, NUM_DIGITS), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if ep % 10 == 0 or ep == EPOCHS:
            exact, pos = evaluate(model, test_loader)
            best = max(best, exact)
            print(f"  ep {ep:3d}/{EPOCHS} [{time.time()-start:.0f}s] "
                  f"exact={exact:.2f}% (best={best:.2f}%)", flush=True)

    exact, pos = evaluate(model, test_loader)
    best = max(best, exact)
    elapsed = time.time() - start

    print(f"\nFinal: exact={exact:.2f}% (best={best:.2f}%) [{elapsed:.0f}s]")
    print("Per-position accuracy:")
    for i in range(seq_len):
        bar = "#" * int(pos[i] / 2)
        print(f"  pos {i}: {pos[i]:5.1f}% {bar}")

    return best, elapsed, n_params


# --- DATA ---
train_ds = ReverseDataset(N_TRAIN, seq_len, seed=42)
test_ds = ReverseDataset(N_TEST, seq_len, seed=9999)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

# --- TRAIN BOTH ---
mem_model = MemoryGRU(hidden_size, seq_len).to(device)
mem_best, mem_time, mem_params = train_and_eval(
    mem_model, "Memory-GRU", train_loader, test_loader
)

van_model = VanillaGRU(hidden_size, seq_len).to(device)
van_best, van_time, van_params = train_and_eval(
    van_model, "Vanilla-GRU", train_loader, test_loader
)

# --- DIAGNOSTICS: show what memory learned ---
print(f"\n{'=' * 60}")
print("MEMORY ACCESS PATTERN (first test example)")
print(f"{'=' * 60}")

mem_model.eval()
with torch.no_grad():
    x_s, y_s = test_ds[0]
    logits, diag = mem_model(x_s.unsqueeze(0).to(device), return_diag=True)
    preds = logits.argmax(dim=2).squeeze(0).cpu()

    print(f"Input:    {x_s.tolist()}")
    print(f"Expected: {y_s.tolist()}")
    print(f"Got:      {preds.tolist()}")
    correct = (preds == y_s).all().item()
    print(f"Match:    {'YES' if correct else 'NO'}")

    print(f"\nENCODING (writing to memory):")
    print(f"  {'step':>4} {'digit':>5} {'wgate':>6}  "
          f"slot weights → peak")
    for t in range(seq_len):
        a = diag["enc_addr"][t]
        wg = diag["enc_wgate"][t]
        slots = " ".join(f"{v:.2f}" for v in a.tolist())
        peak = a.argmax().item()
        print(f"  t={t:2d}  d={x_s[t].item()}  wg={wg:.2f}  [{slots}] → slot {peak}")

    print(f"\nDECODING (reading from memory):")
    print(f"  {'step':>4}  slot weights → peak  pred  exp")
    for t in range(seq_len):
        a = diag["dec_addr"][t]
        slots = " ".join(f"{v:.2f}" for v in a.tolist())
        peak = a.argmax().item()
        mark = "OK" if preds[t].item() == y_s[t].item() else "XX"
        print(f"  t={t:2d}  [{slots}] → slot {peak}  "
              f"pred={preds[t].item()} exp={y_s[t].item()} {mark}")

# --- SUMMARY ---
print(f"\n{'=' * 60}")
print(f"SUMMARY (seq={seq_len}, hidden={hidden_size})")
print(f"{'=' * 60}")
print(f"  Memory-GRU:  {mem_best:6.2f}%  {mem_params:>7,} params  {mem_time:.0f}s")
print(f"  Vanilla-GRU: {van_best:6.2f}%  {van_params:>7,} params  {van_time:.0f}s")
