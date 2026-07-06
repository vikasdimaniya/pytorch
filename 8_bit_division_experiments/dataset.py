import torch
from torch.utils.data import Dataset, DataLoader

N_BITS = 8


def int_to_bits_msb(n, n_bits):
    """Convert integer to bits, MSB first."""
    return [(n >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]


def bits_to_int_msb(bits):
    """Convert MSB-first bit list back to integer."""
    n = 0
    for b in bits:
        n = (n << 1) | int(b)
    return n


class DivisionDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        q = a // b

        bits_a = int_to_bits_msb(a, N_BITS)  # 8 bits, MSB first
        bits_b = int_to_bits_msb(b, N_BITS)  # 8 bits, MSB first
        bits_q = int_to_bits_msb(q, N_BITS)  # quotient fits in 8 bits (since A < 256)

        # For seq2seq: at each step t, input = [a_bit_t] + all_b_bits = 9 values
        # Shape: (8, 9)
        x_seq = []
        for t in range(N_BITS):
            step_input = [bits_a[t]] + bits_b
            x_seq.append(step_input)

        # For flat models: input = all_a_bits + all_b_bits = 16 values
        x_flat = torch.tensor(bits_a + bits_b, dtype=torch.float32)

        x_seq = torch.tensor(x_seq, dtype=torch.float32)    # (8, 9)
        y = torch.tensor(bits_q, dtype=torch.float32)        # (8,)
        return x_flat, x_seq, y


def get_loaders(batch_size=128):
    """Generate all valid pairs where A > B, both in [1, 255]."""
    all_pairs = []
    for b in range(1, 256):
        for a in range(b + 1, 256):
            all_pairs.append((a, b))

    # ~32k pairs total. 80/20 split.
    n = len(all_pairs)
    import random
    random.seed(42)
    random.shuffle(all_pairs)
    split = int(0.8 * n)
    train_pairs = all_pairs[:split]
    test_pairs = all_pairs[split:]

    print(f"Dataset: {len(train_pairs)} train, {len(test_pairs)} test (A > B, both in [1,255])")

    train_loader = DataLoader(DivisionDataset(train_pairs), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(DivisionDataset(test_pairs), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate_seq2seq(model, test_loader, device):
    model.eval()
    total_bits = 0
    correct_bits = 0
    total_samples = 0
    exact_matches = 0
    with torch.no_grad():
        for x_flat, x_seq, y in test_loader:
            x_seq, y = x_seq.to(device), y.to(device)
            outputs = model(x_seq)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_bits += (preds == y).sum().item()
            total_bits += y.numel()
            exact_matches += (preds == y).all(dim=1).sum().item()
            total_samples += y.size(0)
    return 100.0 * correct_bits / total_bits, 100.0 * exact_matches / total_samples


def evaluate_flat(model, test_loader, device):
    model.eval()
    total_bits = 0
    correct_bits = 0
    total_samples = 0
    exact_matches = 0
    with torch.no_grad():
        for x_flat, x_seq, y in test_loader:
            x_flat, y = x_flat.to(device), y.to(device)
            outputs = model(x_flat)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_bits += (preds == y).sum().item()
            total_bits += y.numel()
            exact_matches += (preds == y).all(dim=1).sum().item()
            total_samples += y.size(0)
    return 100.0 * correct_bits / total_bits, 100.0 * exact_matches / total_samples
