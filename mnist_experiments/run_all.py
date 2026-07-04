import subprocess
import sys

MODELS = [
    ("1_linear.py",      "Linear"),
    ("2_mlp.py",         "MLP"),
    ("3_cnn_5x5.py",     "CNN_5x5"),
    ("4_cnn_3x3.py",     "CNN_3x3"),
    ("5_rnn_custom.py",  "RNN_Custom"),
    ("6_rnn_builtin.py", "RNN_Builtin"),
    ("7_lstm.py",        "LSTM"),
    ("8_gru.py",         "GRU"),
    ("9_rnn_4row.py",    "RNN_4Row"),
]


def run_model(script):
    cmd = [sys.executable, script]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    for line in result.stdout.strip().split('\n'):
        if line.startswith("RESULT|"):
            parts = line.split("|")
            return {
                "name": parts[1],
                "params": int(parts[2]),
                "accuracy": float(parts[3]),
                "train_time": float(parts[4]),
            }
    print(f"  ERROR: no RESULT line. stderr: {result.stderr[-300:]}")
    return None


def print_table(results):
    print(f"\n{'='*70}")
    print(f"  MNIST Architecture Comparison (~97% target)")
    print(f"{'='*70}")
    print(f"{'Model':<14} {'Params':>10} {'Accuracy':>10} {'Train Time':>12}")
    print(f"{'-'*14} {'-'*10} {'-'*10} {'-'*12}")
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{r['name']:<14} {r['params']:>10,} {r['accuracy']:>9.2f}% {r['train_time']:>10.2f}s")
    print()


if __name__ == "__main__":
    results = []
    for script, name in MODELS:
        print(f"Running {name}...")
        r = run_model(script)
        if r:
            results.append(r)
            print(f"  -> {r['accuracy']:.2f}%, {r['params']:,} params, {r['train_time']:.2f}s")

    print_table(results)
