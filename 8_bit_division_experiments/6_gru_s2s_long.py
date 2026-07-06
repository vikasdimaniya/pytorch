"""GRU seq2seq with 200 epochs to test if more training pushes toward 100%."""
import torch
import torch.nn as nn
import time
from dataset import get_loaders, evaluate_seq2seq, N_BITS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 128
num_epochs = 200
learning_rate = 0.001

train_loader, test_loader = get_loaders()


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

start_time = time.time()
best_exact = 0
for epoch in range(num_epochs):
    model.train()
    for x_flat, x_seq, y in train_loader:
        x_seq, y = x_seq.to(device), y.to(device)
        outputs = model(x_seq)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        elapsed = time.time() - start_time
        bit_acc, exact_acc = evaluate_seq2seq(model, test_loader, device)
        best_exact = max(best_exact, exact_acc)
        print(f"Epoch {epoch+1}/{num_epochs} [{elapsed:.0f}s], bit={bit_acc:.2f}%, exact={exact_acc:.2f}% (best={best_exact:.2f}%)", flush=True)

train_time = time.time() - start_time
bit_acc, exact_acc = evaluate_seq2seq(model, test_loader, device)
print(f"RESULT|GRU-S2S-200ep|{n_params}|{bit_acc:.2f}|{exact_acc:.2f}|{train_time:.2f}")
