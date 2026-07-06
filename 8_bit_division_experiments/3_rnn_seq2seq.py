"""
Seq2seq RNN for division: mimics long division.
At each step t:
  input = [a_bit_t (1 bit), full_B (8 bits)] = 9 inputs
  output = quotient_bit_t (1 bit)
  hidden state = partial remainder
Processing order: MSB first (natural for long division).
"""
import torch
import torch.nn as nn
import time
from dataset import get_loaders, evaluate_seq2seq, N_BITS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 128
num_epochs = 30
learning_rate = 0.001

train_loader, test_loader = get_loaders()


class RNNSeq2Seq(nn.Module):
    def __init__(self):
        super(RNNSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(9, hidden_size, batch_first=True)  # 1 dividend bit + 8 divisor bits
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, 8, 9)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)          # (batch, 8, hidden)
        bits = self.fc(out).squeeze(2)    # (batch, 8)
        return bits


model = RNNSeq2Seq().to(device)
n_params = sum(p.numel() for p in model.parameters())

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for x_flat, x_seq, y in train_loader:
        x_seq, y = x_seq.to(device), y.to(device)
        outputs = model(x_seq)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        elapsed = time.time() - start_time
        bit_acc, exact_acc = evaluate_seq2seq(model, test_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} [{elapsed:.0f}s], bit={bit_acc:.2f}%, exact={exact_acc:.2f}%", flush=True)

train_time = time.time() - start_time
bit_acc, exact_acc = evaluate_seq2seq(model, test_loader, device)
print(f"RESULT|RNN-S2S|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.2f}")
