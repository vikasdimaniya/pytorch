import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def int_to_bits(n, n_bits=8):
    """Convert integer to list of bits, LSB first."""
    return [(n >> i) & 1 for i in range(n_bits)]


def bits_to_int(bits):
    """Convert list of bits (LSB first) back to integer."""
    return sum(b * (2 ** i) for i, b in enumerate(bits))


class AdditionDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        s = a + b

        bits_a = int_to_bits(a, 8)   # 8 bits, LSB first
        bits_b = int_to_bits(b, 8)   # 8 bits, LSB first
        bits_sum = int_to_bits(s, 9) # 9 bits, LSB first (max 510)

        # flat input: 16 floats (a's 8 bits + b's 8 bits)
        x = torch.tensor(bits_a + bits_b, dtype=torch.float32)
        y = torch.tensor(bits_sum, dtype=torch.float32)
        return x, y


def get_loaders(batch_size=256, train_ratio=0.8):
    all_pairs = [(a, b) for a in range(256) for b in range(256)]
    np.random.seed(42)
    np.random.shuffle(all_pairs)

    split = int(len(all_pairs) * train_ratio)
    train_pairs = all_pairs[:split]
    test_pairs = all_pairs[split:]

    train_loader = DataLoader(AdditionDataset(train_pairs), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(AdditionDataset(test_pairs), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def evaluate(model, test_loader, device, reshape_fn=None):
    """Compute bit accuracy and exact match accuracy."""
    model.eval()
    total_bits = 0
    correct_bits = 0
    total_samples = 0
    exact_matches = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if reshape_fn:
                x = reshape_fn(x)
            outputs = model(x)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            correct_bits += (preds == y).sum().item()
            total_bits += y.numel()

            exact_matches += (preds == y).all(dim=1).sum().item()
            total_samples += y.size(0)

    bit_acc = 100.0 * correct_bits / total_bits
    exact_acc = 100.0 * exact_matches / total_samples
    return bit_acc, exact_acc
