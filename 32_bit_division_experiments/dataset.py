import torch
from torch.utils.data import Dataset, DataLoader
import random

N_BITS = 32
OUT_BITS = 32  # quotient of A // B fits in 32 bits


def int_to_bits(n, n_bits):
    """Convert non-negative integer to list of bits, LSB first."""
    return [(n >> i) & 1 for i in range(n_bits)]


def bits_to_int(bits):
    """Convert LSB-first bit list back to integer."""
    return sum(b * (1 << i) for i, b in enumerate(bits))


class DivisionDataset(Dataset):
    def __init__(self, n_samples, seed=None):
        if seed is not None:
            random.seed(seed)
        max_val = (1 << N_BITS) - 1
        self.data = []
        for _ in range(n_samples):
            a = random.randint(0, max_val)
            b = random.randint(1, max_val)  # never 0
            self.data.append((a, b, a // b))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        a, b, q = self.data[idx]
        bits_a = int_to_bits(a, N_BITS)
        bits_b = int_to_bits(b, N_BITS)
        bits_q = int_to_bits(q, OUT_BITS)

        x = torch.tensor(bits_a + bits_b, dtype=torch.float32)  # 64 inputs
        y = torch.tensor(bits_q, dtype=torch.float32)            # 32 outputs
        return x, y


def get_loaders(batch_size=256, n_train=40000, n_test=10000):
    train_loader = DataLoader(DivisionDataset(n_train, seed=42), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(DivisionDataset(n_test, seed=123), batch_size=batch_size, shuffle=False)
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
