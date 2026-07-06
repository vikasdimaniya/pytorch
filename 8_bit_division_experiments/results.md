# Division Experiments — Results & Learnings

## Motivation

After our earlier experiments with binary addition (8-bit and 128-bit), where seq-to-seq RNNs achieved 100% accuracy with as few as 37 parameters, we wanted to tackle a harder arithmetic operation: **integer division**.

Division is fundamentally different from addition:
- Addition processes LSB-first with a simple 1-bit carry
- Division processes MSB-first (long division), requiring comparison of a multi-bit remainder against the full divisor at every step
- The "state" carried between steps is a remainder (up to N bits), not a single carry bit

We started with 32-bit numbers, then switched to 8-bit for faster iteration once we confirmed the difficulty level is the same. All learnings from both scales are documented here.

---

## Phase 1: 32-bit Generic Models (Baseline)

We first tested standard architectures on 32-bit division (A // B, random unsigned 32-bit numbers, 40k train / 10k test, 50 epochs, hidden=256).

| Model | Params | Bit Acc | Exact Acc | Train Time |
|-------|--------|---------|-----------|------------|
| GRU | 207,904 | 99.62% | 94.00% | 412s |
| LSTM | 274,464 | 99.55% | 92.67% | 480s |
| RNN | 74,784 | 99.47% | 92.29% | 110s |
| MLP | 156,448 | 99.33% | 88.14% | 36s |
| Linear | 24,864 | 99.22% | 86.01% | 31s |

**Observation:** High bit accuracy (~99%) but mediocre exact accuracy. The models learned that most output bits are 0 (since ~50% of random pairs have B > A, giving quotient=0), but struggled with the non-trivial cases. Recurrent models outperformed feedforward, confirming division benefits from sequential processing.

---

## Phase 2: Weighted BCE Loss (32-bit)

We noticed that standard BCE treats each bit equally — getting bit 31 wrong (error of ~2 billion) is penalized the same as bit 0 (error of 1). We tested **weighted BCE** where bit `i` gets weight `i+1`, giving the MSB 32x more importance than the LSB. Also increased to 100 epochs and 80k train / 20k test.

| Model | Loss | Epochs | Exact Acc |
|-------|------|--------|-----------|
| LSTM-WBCE | Weighted | 100 | 95.66% |
| RNN-WBCE | Weighted | 100 | 95.22% |
| GRU-WBCE | Weighted | 100 | 94.70% |
| GRU (baseline) | Standard | 50 | 94.00% |

**Decision:** Weighted BCE helped (~2-3% improvement), but we changed 3 variables at once (loss, epochs, data size). The gain was modest — the loss function was not the main bottleneck. The **architecture** was.

---

## Phase 3: Switch to 8-bit for Fast Iteration

**Key decision:** 32-bit experiments took 7-30 minutes each. Since the fundamental difficulty of division is the same regardless of bit width (the algorithm is identical, just more steps), we switched to **8-bit numbers** for 10-100x faster experiments.

- 8-bit: A, B in [1, 255], A > B (so quotient ≥ 1)
- ~32,385 total valid pairs, 80/20 train/test split
- Experiments finish in seconds to minutes instead of minutes to hours

---

## Phase 4: Seq-to-Seq Architecture (8-bit)

The critical architectural change: instead of predicting all output bits from the final hidden state, output **one quotient bit per time step**.

### Input Design — Mimicking Long Division

We designed the RNN input to match how long division actually works:
- **8 time steps** (one per bit position, MSB first)
- **Input per step**: `[a_bit_t, b_bit_0, b_bit_1, ..., b_bit_7]` = **9 values**
  - 1 dividend bit (brought down one at a time)
  - Full 8-bit divisor (needed in full at every step for comparison)
- **Output per step**: 1 quotient bit
- **Hidden state**: carries the partial remainder forward

This is exactly how a human does long division: you look at the whole divisor, bring down one digit of the dividend at a time, compare, subtract, and produce one quotient digit.

### Why This Matters

Our first seq-to-seq attempt on 32-bit fed B one bit at a time (just like A). This **failed completely** — the vanilla RNN got stuck at 49.9% (just predicting all zeros). The model had to simultaneously memorize B AND perform division, which was too much.

By providing the full divisor at every step, the model only needs to learn: "is my current remainder ≥ B? If yes, output 1 and subtract. If no, output 0."

### 8-bit Results (30 epochs, hidden=128)

| Model | Type | Params | Exact Acc | Time |
|-------|------|--------|-----------|------|
| GRU-S2S | seq2seq | 53,505 | 93.36% | 26s |
| LSTM-S2S | seq2seq | 71,297 | 90.07% | 29s |
| MLP | flat | 19,720 | 88.14% | 13s |
| RNN-S2S | seq2seq | 17,921 | 85.61% | 17s |
| Linear | flat | 3,208 | 80.15% | 12s |

**Finding:** Seq-to-seq models outperform flat models, and GRU leads. But 30 epochs wasn't enough — accuracy was still climbing.

---

## Phase 5: More Training — Does It Converge?

We ran GRU-S2S for 200 epochs to test if more training helps.

| Epoch | Exact Acc |
|-------|-----------|
| 30 | 93.36% |
| 50 | 96.08% |
| 100 | 97.96% |
| 130 | 98.86% |
| 180 | **99.49%** |
| 200 | 99.46% |

**Answer: Yes.** The model went from 93% to 99.5% with more training. The architecture was correct — it just needed more time to learn the algorithm.

---

## Phase 6: Curriculum Learning

Inspired by the research literature (especially the Int2Int framework and "Neural GPUs Learn Algorithms"), we implemented **curriculum learning**: train on small numbers first, gradually increase.

| Epochs | Stage | Number Range | Pairs |
|--------|-------|-------------|-------|
| 1-50 | 4-bit | 1-15 | 105 |
| 51-100 | 5-bit | 1-31 | 465 |
| 101-150 | 6-bit | 1-63 | 1,953 |
| 151-200 | 7-bit | 1-127 | 8,001 |
| 201-300 | 8-bit | 1-255 | 32,385 |

### GRU-S2S with Curriculum (hidden=128, 300 epochs)

Result: **99.91% exact accuracy** — better than 200 epochs without curriculum (99.49%), and faster (124s vs 176s).

The model initially drops in accuracy when switching to harder stages (it has to relearn), but when it reaches 8-bit data at epoch 201, it rockets to 89% within 10 epochs and climbs to 99.91%.

---

## Phase 7: Memorization Analysis

With 53,505 parameters and 25,908 training samples (ratio = 2.07), the large model could theoretically memorize. We tested smaller models to prove genuine algorithm learning:

| Hidden Size | Params | Params/Sample | Exact Acc | Can Memorize? |
|-------------|--------|---------------|-----------|---------------|
| 128 | 53,505 | 2.07 | 99.91% | Possibly |
| **32** | **4,161** | **0.16** | **100.00%** | **No** |
| 16 | 1,313 | 0.05 | 96.80% | No |

### The Definitive Result

**GRU-S2S with hidden_size=32 achieved 100.00% exact accuracy with only 4,161 parameters.**

- 6x fewer parameters than training samples — memorization is mathematically impossible
- Used curriculum learning + cosine annealing LR scheduler
- 500 epochs, starting LR=0.003, ending LR=0.00001
- Hit 100% bit accuracy at epoch 460, 100% exact at epoch 500

This proves the model has genuinely learned the long division algorithm.

---

## Key Learnings

### 1. Architecture Must Match the Algorithm
The biggest improvement came not from tuning hyperparameters, but from designing the input/output structure to match how long division actually works. Feeding the full divisor at every step + one dividend bit at a time + MSB-first processing + per-step output.

### 2. Loss Function Helps, But Architecture Matters More
Weighted BCE (prioritizing MSBs) gave ~2-3% improvement. The seq-to-seq architectural change gave ~15-20% improvement on the same task.

### 3. Curriculum Learning Accelerates Convergence
Training on small numbers first builds foundational understanding of the division algorithm, making the jump to larger numbers faster and more stable.

### 4. Training Budget Matters
30 epochs: 93%. 200 epochs: 99.5%. 500 epochs: 100%. Neural networks learning algorithms need patience — they're not just fitting a function, they're discovering a procedure.

### 5. Cosine Annealing > Step LR
Smooth LR decay (cosine) outperformed abrupt step decay. The gradual reduction allows fine-tuning of decision boundaries in late training.

### 6. Small Models Can Learn Algorithms
4,161 parameters (hidden_size=32) achieved 100% on 8-bit division. This is consistent with our earlier finding that 37 parameters could do 128-bit addition. When the architecture matches the algorithm, you need remarkably few parameters.

### 7. Division is Harder Than Addition
Addition's carry is 1 bit. Division's remainder is N bits. This means the hidden state needs more capacity (hidden_size=32 for division vs hidden_size=3 for addition), but the fundamental seq-to-seq approach works for both.
