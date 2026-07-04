import torch
from torch.utils.data import Dataset, DataLoader
import random

N_BITS = 128
OUT_BITS = 129  # max sum needs 129 bits


def int_to_bits(n, n_bits):
    """Convert integer to list of bits, LSB first."""
    return [(n >> i) & 1 for i in range(n_bits)]


def bits_to_int(bits):
    """Convert list of bits (LSB first) back to integer."""
    return sum(b * (2 ** i) for i, b in enumerate(bits))


class AdditionDataset(Dataset):
    def __init__(self, n_samples, seed=None):
        if seed is not None:
            random.seed(seed)
        max_val = (1 << N_BITS) - 1
        self.pairs = [(random.randint(0, max_val), random.randint(0, max_val))
                      for _ in range(n_samples)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        s = a + b

        bits_a = int_to_bits(a, N_BITS)
        bits_b = int_to_bits(b, N_BITS)
        bits_sum = int_to_bits(s, OUT_BITS)

        x = torch.tensor(bits_a + bits_b, dtype=torch.float32)
        y = torch.tensor(bits_sum, dtype=torch.float32)
        return x, y


def get_loaders(batch_size=256, n_train=20000, n_test=5000):
    train_loader = DataLoader(AdditionDataset(n_train, seed=42), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(AdditionDataset(n_test, seed=123), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate(model, test_loader, device, reshape_fn=None):
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
