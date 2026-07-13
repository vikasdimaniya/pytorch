import torch
from torch.utils.data import Dataset, DataLoader
import random

N_BITS = 32
OUT_BITS = 64  # enough for multiplication (32-bit * 32-bit = up to 64 bits)
OPS = {0: 'add', 1: 'sub', 2: 'mul', 3: 'div'}


def int_to_bits(n, n_bits):
    """Convert non-negative integer to list of bits, LSB first."""
    return [(n >> i) & 1 for i in range(n_bits)]


def compute(op, a, b):
    """Compute result for given operation. All results are non-negative."""
    if op == 0:    # add
        return a + b
    elif op == 1:  # sub (absolute difference)
        return abs(a - b)
    elif op == 2:  # mul
        return a * b
    elif op == 3:  # div (integer division, B is never 0)
        return a // b


class ArithmeticDataset(Dataset):
    def __init__(self, n_samples, seed=None):
        if seed is not None:
            random.seed(seed)
        max_val = (1 << N_BITS) - 1
        self.data = []
        per_op = n_samples // 4
        for op in range(4):
            for _ in range(per_op):
                a = random.randint(0, max_val)
                b = random.randint(1 if op == 3 else 0, max_val)  # avoid div by 0
                self.data.append((op, a, b))
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        op, a, b = self.data[idx]
        result = compute(op, a, b)

        op_bits = int_to_bits(op, 2)         # 2 bits for operation
        bits_a = int_to_bits(a, N_BITS)      # 32 bits
        bits_b = int_to_bits(b, N_BITS)      # 32 bits
        bits_result = int_to_bits(result, OUT_BITS)  # 64 bits

        x = torch.tensor(op_bits + bits_a + bits_b, dtype=torch.float32)  # 66 inputs
        y = torch.tensor(bits_result, dtype=torch.float32)                # 64 outputs
        return x, y


def get_loaders(batch_size=256, n_train=40000, n_test=10000):
    train_loader = DataLoader(ArithmeticDataset(n_train, seed=42), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ArithmeticDataset(n_test, seed=123), batch_size=batch_size, shuffle=False)
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
