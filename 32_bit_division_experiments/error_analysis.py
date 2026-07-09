"""
Error analysis: diagnose WHERE and WHY the model fails.
Loads best checkpoint, runs exhaustive test, generates visualizations.
"""
import torch
import torch.nn as nn
import numpy as np
import random
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

torch.set_num_threads(8)
device = torch.device("cpu")

N_BITS = 32
hidden_size = 256
CHECKPOINT_PATH = "best_model_h256.pt"
OUT_DIR = "error_analysis_plots"
os.makedirs(OUT_DIR, exist_ok=True)


def int_to_bits_msb(n, n_bits):
    return [(n >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]

def bits_to_int_msb(bits):
    n = 0
    for b in bits:
        n = (n << 1) | int(b)
    return n

def bit_length(n):
    if n == 0:
        return 0
    return n.bit_length()


class GRUSeq2Seq(nn.Module):
    def __init__(self):
        super(GRUSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(33, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out).squeeze(2)


# Load model
model = GRUSeq2Seq().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
model.eval()
print("Model loaded from checkpoint")


# Generate large test set: 50k samples, mixed across all bit sizes
def sample_pairs(max_bits, n_samples, seed):
    max_val = (1 << max_bits) - 1
    rng = random.Random(seed)
    if max_bits <= 8:
        pairs = [(a, b) for b in range(1, max_val + 1) for a in range(b + 1, max_val + 1)]
        rng.shuffle(pairs)
        return pairs[:n_samples]
    pairs = set()
    while len(pairs) < n_samples:
        b = rng.randint(1, max_val)
        a = rng.randint(b + 1, max(b + 2, max_val))
        if a <= max_val:
            pairs.add((a, b))
    return list(pairs)

print("Generating test data...")
all_pairs = []
for max_bits in [4, 8, 12, 16, 20, 24, 28, 32]:
    seed = 55555 + max_bits  # different seed from training
    pairs = sample_pairs(max_bits, 6000, seed)
    all_pairs.extend(pairs)
random.Random(777).shuffle(all_pairs)
print(f"Test set: {len(all_pairs)} pairs")


# Run all pairs through the model, collect results
print("Running inference...")
results = []
BATCH = 512

for i in range(0, len(all_pairs), BATCH):
    batch_pairs = all_pairs[i:i+BATCH]
    x_list = []
    for a, b in batch_pairs:
        bits_a = int_to_bits_msb(a, N_BITS)
        bits_b = int_to_bits_msb(b, N_BITS)
        x_seq = [[bits_a[t]] + bits_b for t in range(N_BITS)]
        x_list.append(x_seq)

    x = torch.tensor(x_list, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
        preds = (torch.sigmoid(logits) > 0.5).float()

    for j, (a, b) in enumerate(batch_pairs):
        q_true = a // b
        bits_true = int_to_bits_msb(q_true, N_BITS)
        bits_pred = preds[j].cpu().numpy().astype(int).tolist()
        q_pred = bits_to_int_msb(bits_pred)

        bit_errors = [1 if bits_true[k] != bits_pred[k] else 0 for k in range(N_BITS)]
        is_exact = (q_true == q_pred)

        results.append({
            'a': a, 'b': b,
            'q_true': q_true, 'q_pred': q_pred,
            'a_bits': bit_length(a), 'b_bits': bit_length(b), 'q_bits': bit_length(q_true),
            'ratio': a / b,
            'bit_errors': bit_errors,
            'n_bit_errors': sum(bit_errors),
            'is_exact': is_exact,
            'numerical_error': abs(q_true - q_pred),
        })

total = len(results)
n_correct = sum(1 for r in results if r['is_exact'])
n_wrong = total - n_correct
print(f"\nOverall: {n_correct}/{total} correct ({100*n_correct/total:.2f}%)")
print(f"Errors: {n_wrong} samples\n")

errors = [r for r in results if not r['is_exact']]


# ============================================================
# PLOT 1: Error rate by bit position (MSB=0, LSB=31)
# ============================================================
print("Plot 1: Error rate by bit position...")
bit_error_counts = np.zeros(N_BITS)
for r in results:
    for k in range(N_BITS):
        bit_error_counts[k] += r['bit_errors'][k]

bit_error_rate = bit_error_counts / total * 100

fig, ax = plt.subplots(figsize=(14, 5))
positions = np.arange(N_BITS)
colors = ['#e74c3c' if r > np.median(bit_error_rate[bit_error_rate > 0]) else '#3498db'
          for r in bit_error_rate]
ax.bar(positions, bit_error_rate, color=colors, edgecolor='black', linewidth=0.3)
ax.set_xlabel('Bit Position (0=MSB, 31=LSB)', fontsize=12)
ax.set_ylabel('Error Rate (%)', fontsize=12)
ax.set_title('Error Rate by Output Bit Position', fontsize=14)
ax.set_xticks(range(0, 32, 2))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/1_error_by_bit_position.png', dpi=150)
plt.close()
print(f"  Top error bits: {np.argsort(bit_error_rate)[-5:][::-1]} with rates {np.sort(bit_error_rate)[-5:][::-1]}")


# ============================================================
# PLOT 2: Error rate by quotient bit-length
# ============================================================
print("Plot 2: Error rate by quotient bit-length...")
by_qbits = defaultdict(lambda: {'total': 0, 'errors': 0})
for r in results:
    qb = r['q_bits']
    by_qbits[qb]['total'] += 1
    if not r['is_exact']:
        by_qbits[qb]['errors'] += 1

q_bit_lengths = sorted(by_qbits.keys())
q_error_rates = [100 * by_qbits[qb]['errors'] / by_qbits[qb]['total'] for qb in q_bit_lengths]
q_counts = [by_qbits[qb]['total'] for qb in q_bit_lengths]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax1.bar(q_bit_lengths, q_error_rates, color='#e74c3c', edgecolor='black', linewidth=0.3)
ax1.set_ylabel('Error Rate (%)', fontsize=12)
ax1.set_title('Error Rate by Quotient Bit-Length', fontsize=14)
ax1.grid(axis='y', alpha=0.3)

ax2.bar(q_bit_lengths, q_counts, color='#2ecc71', edgecolor='black', linewidth=0.3)
ax2.set_xlabel('Quotient Bit-Length', fontsize=12)
ax2.set_ylabel('Sample Count', fontsize=12)
ax2.set_title('Sample Distribution by Quotient Bit-Length', fontsize=14)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/2_error_by_quotient_bitlength.png', dpi=150)
plt.close()


# ============================================================
# PLOT 3: Error rate by dividend bit-length vs divisor bit-length
# ============================================================
print("Plot 3: Error heatmap (A bits vs B bits)...")
heatmap_err = np.zeros((33, 33))
heatmap_count = np.zeros((33, 33))
for r in results:
    ab, bb = r['a_bits'], r['b_bits']
    heatmap_count[ab][bb] += 1
    if not r['is_exact']:
        heatmap_err[ab][bb] += 1

with np.errstate(divide='ignore', invalid='ignore'):
    heatmap_rate = np.where(heatmap_count > 0, 100 * heatmap_err / heatmap_count, np.nan)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(heatmap_rate[1:, 1:], cmap='RdYlGn_r', aspect='auto',
               origin='lower', vmin=0, vmax=20)
ax.set_xlabel('Divisor (B) Bit-Length', fontsize=12)
ax.set_ylabel('Dividend (A) Bit-Length', fontsize=12)
ax.set_title('Error Rate (%) by A Bit-Length vs B Bit-Length', fontsize=14)
plt.colorbar(im, label='Error Rate (%)')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/3_error_heatmap_a_vs_b_bits.png', dpi=150)
plt.close()


# ============================================================
# PLOT 4: Error rate by A/B ratio (quotient magnitude)
# ============================================================
print("Plot 4: Error rate by A/B ratio...")
ratio_bins = [1, 2, 4, 8, 16, 64, 256, 1024, 2**16, 2**24, 2**32]
ratio_labels = ['1-2', '2-4', '4-8', '8-16', '16-64', '64-256',
                '256-1K', '1K-64K', '64K-16M', '16M-4B']
bin_errors = [0] * len(ratio_labels)
bin_totals = [0] * len(ratio_labels)

for r in results:
    ratio = r['ratio']
    for k in range(len(ratio_bins) - 1):
        if ratio_bins[k] <= ratio < ratio_bins[k + 1]:
            bin_totals[k] += 1
            if not r['is_exact']:
                bin_errors[k] += 1
            break

bin_rates = [100 * e / t if t > 0 else 0 for e, t in zip(bin_errors, bin_totals)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
x = np.arange(len(ratio_labels))
ax1.bar(x, bin_rates, color='#e74c3c', edgecolor='black', linewidth=0.3)
ax1.set_xticks(x)
ax1.set_xticklabels(ratio_labels, rotation=45, ha='right')
ax1.set_ylabel('Error Rate (%)', fontsize=12)
ax1.set_title('Error Rate by Quotient Range (A/B ratio)', fontsize=14)
ax1.grid(axis='y', alpha=0.3)

ax2.bar(x, bin_totals, color='#3498db', edgecolor='black', linewidth=0.3)
ax2.set_xticks(x)
ax2.set_xticklabels(ratio_labels, rotation=45, ha='right')
ax2.set_ylabel('Sample Count', fontsize=12)
ax2.set_title('Sample Distribution by Quotient Range', fontsize=14)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/4_error_by_ratio.png', dpi=150)
plt.close()


# ============================================================
# PLOT 5: Numerical error magnitude distribution
# ============================================================
print("Plot 5: Numerical error distribution...")
if errors:
    num_errors = [r['numerical_error'] for r in errors]
    log_errors = [np.log2(e + 1) for e in num_errors]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(log_errors, bins=50, color='#e74c3c', edgecolor='black', linewidth=0.3)
    ax1.set_xlabel('log₂(|error| + 1)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Numerical Error Magnitude', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)

    n_bit_errs = [r['n_bit_errors'] for r in errors]
    ax2.hist(n_bit_errs, bins=range(1, max(n_bit_errs) + 2), color='#9b59b6',
             edgecolor='black', linewidth=0.3)
    ax2.set_xlabel('Number of Wrong Bits', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Bit Error Count per Wrong Answer', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/5_error_magnitude.png', dpi=150)
    plt.close()


# ============================================================
# PLOT 6: Error rate by divisor characteristics (near powers of 2)
# ============================================================
print("Plot 6: Near-power-of-2 analysis...")
def distance_to_nearest_pow2(n):
    if n <= 0:
        return 0
    bl = n.bit_length()
    lower = 1 << (bl - 1)
    upper = 1 << bl
    return min(n - lower, upper - n)

pow2_bins = defaultdict(lambda: {'total': 0, 'errors': 0})
for r in results:
    dist = distance_to_nearest_pow2(r['b'])
    b_bl = r['b_bits']
    if b_bl == 0:
        continue
    normalized_dist = dist / (1 << (b_bl - 1))  # 0 = exact power of 2, 1 = max distance
    bucket = round(normalized_dist, 1)
    pow2_bins[bucket]['total'] += 1
    if not r['is_exact']:
        pow2_bins[bucket]['errors'] += 1

buckets = sorted(pow2_bins.keys())
rates = [100 * pow2_bins[b]['errors'] / pow2_bins[b]['total'] for b in buckets]
counts = [pow2_bins[b]['total'] for b in buckets]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
ax1.bar(range(len(buckets)), rates, color='#f39c12', edgecolor='black', linewidth=0.3)
ax1.set_xticks(range(len(buckets)))
ax1.set_xticklabels([f'{b:.1f}' for b in buckets], fontsize=8)
ax1.set_xlabel('Normalized Distance to Nearest Power of 2 (0=exact, 1=max)', fontsize=11)
ax1.set_ylabel('Error Rate (%)', fontsize=12)
ax1.set_title('Error Rate by Divisor Distance to Nearest Power of 2', fontsize=14)
ax1.grid(axis='y', alpha=0.3)

ax2.bar(range(len(buckets)), counts, color='#1abc9c', edgecolor='black', linewidth=0.3)
ax2.set_xticks(range(len(buckets)))
ax2.set_xticklabels([f'{b:.1f}' for b in buckets], fontsize=8)
ax2.set_xlabel('Normalized Distance to Nearest Power of 2', fontsize=11)
ax2.set_ylabel('Sample Count', fontsize=12)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/6_near_power_of_2.png', dpi=150)
plt.close()


# ============================================================
# PLOT 7: Scatter of errors — log(A) vs log(B)
# ============================================================
print("Plot 7: Error scatter (log A vs log B)...")
fig, ax = plt.subplots(figsize=(10, 10))

correct_a = [np.log2(r['a'] + 1) for r in results if r['is_exact']]
correct_b = [np.log2(r['b'] + 1) for r in results if r['is_exact']]
error_a = [np.log2(r['a'] + 1) for r in errors]
error_b = [np.log2(r['b'] + 1) for r in errors]

ax.scatter(correct_b, correct_a, c='#2ecc71', alpha=0.1, s=3, label=f'Correct ({n_correct})')
ax.scatter(error_b, error_a, c='#e74c3c', alpha=0.6, s=10, label=f'Wrong ({n_wrong})')
ax.plot([0, 32], [0, 32], 'k--', alpha=0.3, label='A = B')
ax.set_xlabel('log₂(B) — Divisor', fontsize=12)
ax.set_ylabel('log₂(A) — Dividend', fontsize=12)
ax.set_title('Error Locations: log(A) vs log(B)', fontsize=14)
ax.legend(fontsize=11)
ax.set_xlim(0, 33)
ax.set_ylim(0, 33)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/7_error_scatter.png', dpi=150)
plt.close()


# ============================================================
# Print summary statistics
# ============================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"\nTotal: {total} samples, {n_wrong} errors ({100*n_wrong/total:.2f}%)")

if errors:
    print(f"\nBit position error rates (MSB=0, LSB=31):")
    for i in range(N_BITS):
        if bit_error_rate[i] > 0.01:
            print(f"  Bit {i:2d} (value 2^{31-i:2d}): {bit_error_rate[i]:.3f}%")

    print(f"\nError by quotient bit-length:")
    for qb in q_bit_lengths:
        if by_qbits[qb]['total'] >= 10:
            rate = 100 * by_qbits[qb]['errors'] / by_qbits[qb]['total']
            print(f"  Q={qb:2d} bits: {rate:5.1f}% error ({by_qbits[qb]['errors']}/{by_qbits[qb]['total']})")

    print(f"\nError by A/B ratio bucket:")
    for label, rate, tot in zip(ratio_labels, bin_rates, bin_totals):
        if tot > 0:
            print(f"  ratio {label:>10s}: {rate:5.1f}% error ({tot} samples)")

    avg_num_err = np.mean([r['numerical_error'] for r in errors])
    med_num_err = np.median([r['numerical_error'] for r in errors])
    print(f"\nNumerical error stats (wrong answers only):")
    print(f"  Mean: {avg_num_err:.0f}")
    print(f"  Median: {med_num_err:.0f}")
    print(f"  Max: {max(r['numerical_error'] for r in errors)}")

print(f"\nPlots saved to: {os.path.abspath(OUT_DIR)}/")
