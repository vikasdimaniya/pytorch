import subprocess
import sys

MODELS = [
    ("1_linear.py",  "Linear"),
    ("2_mlp.py",     "MLP"),
    ("3_rnn.py",     "RNN"),
    ("4_lstm.py",    "LSTM"),
    ("5_gru.py",     "GRU"),
]


def run_model(script):
    cmd = [sys.executable, script]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    for line in result.stdout.strip().split('\n'):
        if line.startswith("RESULT|"):
            parts = line.split("|")
            return {
                "name": parts[1],
                "params": int(parts[2]),
                "bit_acc": float(parts[3]),
                "exact_acc": float(parts[4]),
                "train_time": float(parts[5]),
            }
    print(f"  ERROR: no RESULT line. stderr: {result.stderr[-500:]}")
    return None


def print_table(results):
    print(f"\n{'='*75}")
    print(f"  32-bit Division — 50 epochs, hidden=256")
    print(f"{'='*75}")
    print(f"{'Model':<10} {'Params':>8} {'Bit Acc':>10} {'Exact Acc':>11} {'Train Time':>12}")
    print(f"{'-'*10} {'-'*8} {'-'*10} {'-'*11} {'-'*12}")
    for r in sorted(results, key=lambda x: x['exact_acc'], reverse=True):
        print(f"{r['name']:<10} {r['params']:>8,} {r['bit_acc']:>9.2f}% {r['exact_acc']:>10.2f}% {r['train_time']:>10.2f}s")
    print()


if __name__ == "__main__":
    results = []
    for script, name in MODELS:
        print(f"Running {name}...")
        r = run_model(script)
        if r:
            results.append(r)
            print(f"  -> bit={r['bit_acc']:.2f}%, exact={r['exact_acc']:.2f}%, {r['params']:,} params, {r['train_time']:.2f}s")

    print_table(results)
