"""
Grid runner: runs all hidden_size x seq_len combinations and prints summary table.
Runs experiments sequentially to avoid CPU contention.
"""
import subprocess
import json
import sys
import os

HIDDEN_SIZES = [32, 64, 128, 256]
SEQ_LENS = [16, 32, 64, 128]

script = os.path.join(os.path.dirname(__file__), "reverse_gru.py")
results = {}

total = len(HIDDEN_SIZES) * len(SEQ_LENS)
done = 0

for h in HIDDEN_SIZES:
    for s in SEQ_LENS:
        done += 1
        print(f"\n{'='*60}")
        print(f"[{done}/{total}] Running h={h}, seq={s}")
        print(f"{'='*60}")

        proc = subprocess.run(
            [sys.executable, script, str(h), str(s)],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)

        for line in proc.stdout.split("\n"):
            if line.startswith("RESULT_JSON|"):
                data = json.loads(line.split("|", 1)[1])
                results[(h, s)] = data

# Print summary table
print("\n" + "=" * 70)
print("SUMMARY: Best Exact Accuracy (%)")
print("=" * 70)

label = r"h \ seq"
header = f"{label:>10}" + "".join(f"{s:>10}" for s in SEQ_LENS)
print(header)
print("-" * len(header))

for h in HIDDEN_SIZES:
    row = f"{h:>10}"
    for s in SEQ_LENS:
        key = (h, s)
        if key in results:
            row += f"{results[key]['best_exact']:>9.1f}%"
        else:
            row += f"{'ERR':>10}"
    print(row)

print()

# Print parameter counts
print("Parameter counts:")
for h in HIDDEN_SIZES:
    for s in SEQ_LENS:
        key = (h, s)
        if key in results:
            print(f"  h={h:>3}, seq={s:>3}: {results[key]['params']:>8,} params, {results[key]['time']:.0f}s")

# Print per-position accuracy for h=32, seq=32
key = (32, 32)
if key in results:
    print(f"\nPer-position accuracy for h=32, seq=32:")
    pos = results[key]["pos_acc"]
    for i, acc in enumerate(pos):
        bar = "#" * int(acc / 2)
        label = f"(input[{31-i}])" if len(pos) == 32 else ""
        print(f"  out[{i:2d}] {label}: {acc:5.1f}% {bar}")
