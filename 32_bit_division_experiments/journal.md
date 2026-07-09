# 32-Bit Division Experiments — Journal

## Goal

Teach a neural network to perform integer division on 32-bit numbers.
Input: dividend A, divisor B (both 32-bit, A > B).
Output: quotient Q = A // B (32-bit).

---

## Phase 1: Flat Models (scripts 1–5)

**Approach:** Feed all 64 input bits (32 for A + 32 for B) as a flat vector. Output all 32 quotient bits at once. Tried Linear, MLP, RNN, LSTM, GRU.

**Result:** All models stuck around 50% bit accuracy — essentially predicting all zeros. Division requires sequential comparison-subtraction steps that flat models can't represent.

**Lesson:** Division is fundamentally sequential. A flat input→output mapping can't learn the algorithm.

---

## Phase 2: Weighted BCE Loss (scripts 6–10)

**Hypothesis:** Maybe standard BCE treats all bits equally, but higher-order bits matter more numerically. Weight the loss: `bit_weights = [i + 1 for i in range(32)]`.

**Result:** Marginal improvement. The loss function wasn't the bottleneck — the architecture was.

**Lesson:** Better loss functions can't compensate for an architecture that doesn't match the problem structure.

---

## Phase 3: Seq-to-Seq (scripts 11–13)

**Key insight:** Long division works MSB-to-LSB: at each step, you look at the next bit of A, compare the running remainder against B, and output one quotient bit. This maps perfectly to a seq-to-seq RNN.

**Design:**
- 32 time steps (one per bit of A, MSB first)
- Input at each step: `[1 bit of A] + [all 32 bits of B]` = 33 features
- Output at each step: 1 quotient bit
- The hidden state tracks the running remainder

**Why this input design:** The model needs ALL of B at every step to do the "is remainder ≥ B?" comparison. Feeding B one bit at a time (like we did for addition) doesn't work for division because you need the full divisor for each comparison.

**Result:** RNN/LSTM/GRU all showed meaningful learning for the first time. GRU performed best. But accuracy was still limited on 32-bit — the model struggled to maintain precise remainder tracking across all 32 steps.

**Lesson:** Architecture must match the algorithm. Seq-to-seq with full-B input mimics long division perfectly.

---

## Phase 4: Curriculum Learning (scripts 14–17)

**Problem:** Training directly on 32-bit division was too hard. The model needed to learn the division algorithm on simpler cases first.

**Approach:** Start with 2-bit numbers, then 4, 8, 16, and finally 32-bit. Gradually increase complexity.

### Script 14: GRU h=64, 7-stage curriculum
- Stages: 8→12→16→20→24→28→32 bits
- Result: **89.23% exact** on 32-bit (first strong result)

### Script 15: GRU h=128, finer curriculum from 2-bit
- More stages, cosine annealing over total epochs
- Problem: **catastrophic forgetting** — when training moved to 32-bit data, the model forgot smaller numbers
- Problem: **LR decay** — cosine annealing over total epochs meant LR was too low by the time hard 32-bit data arrived
- Peak: 45.47%, then decayed

### Script 16: Per-stage LR warm restarts
- Fresh optimizer + scheduler for each curriculum stage
- Crashed due to MPS (Apple GPU) issues

### Script 17: CPU optimization, constant LR for final stage
- Also crashed on MPS. Switched to CPU with `torch.set_num_threads(8)`

**Lessons:**
- Curriculum learning is essential for multi-step algorithmic tasks
- Cosine annealing over total epochs causes LR to be too low for later (harder) stages
- Need mixed-difficulty training data in later stages to prevent catastrophic forgetting
- MPS (Apple GPU) is unstable for this workload — CPU is more reliable

---

## Phase 5: Cyclic LR + Checkpointing (scripts 18–19)

**Breakthrough strategy — 3-phase training:**

1. **Phase 1 — Curriculum warmup:** Quick progression through 2→4→8→16 bit stages with Adam lr=0.001
2. **Phase 2 — Cyclic LR on mixed data:** `CyclicLR` with `triangular2` mode (decaying amplitude). Mixed training data from all bit sizes. This explores broadly while the cyclic peaks discover new solutions.
3. **Phase 3 — Cosine fine-tune:** Load best checkpoint from Phase 2, fine-tune with cosine annealing to squeeze out remaining accuracy.

**Key decisions:**
- Mixed training data (samples from 4/8/12/16/20/24/28/32-bit ranges) prevents catastrophic forgetting
- CyclicLR with triangular2 provides "broad exploration with decay" — high LR peaks find new basins, low LR valleys let the model settle
- Saving best checkpoint + reloading for fine-tune prevents losing good solutions during exploration

### Script 18: h=128 → **97.18% exact**
### Script 19: h=256 → **97.71% exact**
### saved_model_post_training.py: h=256, +200 more epochs → **97.90% exact**

**Lesson:** The 3-phase approach (curriculum → cyclic exploration → cosine refinement) is highly effective. CyclicLR's exploration prevents getting trapped in local minima.

---

## Phase 6: Error Analysis (error_analysis.py)

**97.90% exact sounded great — but was it?**

Generated 42K test samples across all difficulty levels and analyzed where errors land.

### Finding 1: Errors concentrated at LSB
Bit 31 (LSB): 3.5% error rate. Bit 0 (MSB): ~0%. The model gets the high-order bits right but struggles with the final precision bits — exactly the carry/remainder chain problem.

### Finding 2: Error rate scales with quotient bit-length
| Quotient bits | Error Rate |
|---|---|
| 1 (A ≈ B) | 0.4% |
| 2 | 4% |
| 3 | 9% |
| 4 | 17% |
| 5 | **43%** |
| 6-10 | **70-100%** |

The 97.90% accuracy was **misleading** — random sampling of (A, B) pairs gives ~67% of cases with Q=1 (trivially easy). On a balanced test set, the real accuracy was ~20-27%.

### Finding 3: Median numerical error = 1
Most wrong answers are off-by-one. The model approximates well but can't always get the final carry chain right.

### Finding 4: Near-power-of-2 divisors
No significant effect. The error pattern is about quotient complexity, not divisor characteristics.

**Lesson:** Always test with a balanced distribution. Accuracy on random samples can be wildly misleading when difficulty is non-uniform.

---

## Phase 7: Balanced Training + Capacity Experiments (scripts 20–22)

### Balanced sampling
Instead of random (A, B) pairs (dominated by Q=1), sample equal numbers per quotient bit-length. This forces the model to actually learn hard cases.

**Decision:** Generate training pairs by picking a target quotient Q, then constructing A = Q × B + remainder. This guarantees controlled difficulty distribution.

### Capacity comparison (balanced test, ~10 min runs)
| Model | Params | Exact Acc (balanced) |
|---|---|---|
| h=256 | 223K | 20.60% |
| h=512 | 841K | **27.36%** |

h=512 was better across every quotient bit-length. The cliff shifted right by ~1 bit.

### Larger models (h=1024, h=2048)
Tried h=1024 (3.3M params) and h=2048 (12.8M params). Both performed **worse** because larger models need more training epochs to converge, and within a fixed time budget they got far fewer iterations.

**Lesson:** Within a practical time budget, h=512 is the sweet spot. Bigger models are slower per epoch and can't converge in time. The bottleneck is training time, not model capacity.

---

## Phase 8: Hard-Example Mining (script 21)

**Approach:** Oversample the quotient bit-lengths where the model struggles (Q=3-8, the "struggle zone") with 3-4x weight.

**Result:** Extreme oversampling (4x) actually hurt — the model didn't get enough easy examples to build a foundation. A gentler 2x oversampling worked better.

**Lesson:** Hard-example mining helps, but you can't starve the model of easy cases. It needs a foundation of simple examples to build upon.

---

## Phase 9: Architecture Experiments (scripts 23–24)

### 2-layer GRU h=512 + dropout vs 1-layer GRU h=512
Both trained for 1 hour with balanced data.

| Model | Params | Exact Acc | Epochs |
|---|---|---|---|
| 1-layer GRU | 841K | **22.77%** | 65 |
| 2-layer GRU + dropout | 2.4M | 9.75% | 35 |

**2-layer was worse.** Same problem as larger hidden sizes: 3x more parameters → slower per epoch → fewer total epochs → can't converge.

**Lesson:** Architectural depth (more layers) doesn't help when the bottleneck is sequential carry chain length, not per-step representation power.

---

## Phase 10: Extended Training (scripts 25–28)

### Can more training time break the ceiling?

Resumed the best h=512 1-layer GRU checkpoint repeatedly, each time for 1 hour with fresh training data (different random seeds to avoid overfitting).

| Cumulative Hours | Exact Acc (balanced) | Gain |
|---|---|---|
| 1h | 22.77% | — |
| 2h | 38.26% | +15.5% |
| 3h | 55.79% | +17.5% |
| **4h** | **56.96%** | **+1.2%** |

### Final per-quotient-bit-length accuracy (best model)

| Q bits | Accuracy | Meaning |
|---|---|---|
| 1 | 98.2% | Trivial comparison — nearly perfect |
| 2 | 92.4% | 1-step division — strong |
| 3 | 92.9% | 2-step division — strong |
| 4 | 89.5% | 3-step division — good |
| 5 | 78.0% | 4-step division — decent |
| 6 | 52.3% | 5-step division — coin flip |
| 7 | 28.8% | 6-step division — struggling |
| 8 | 14.0% | 7-step division — mostly guessing |
| 9 | 7.2% | 8-step division — near random |
| 10 | 2.6% | 9-step division — random |

**The model plateaued at ~57%.** Hour 4 gained only +1.2% while hours 2-3 gained +15-17% each. The easy cases (Q=1-4) slightly degraded while hard cases (Q=5-7) crept up — a classic sign of hitting the architecture's ceiling.

**Lesson:** More training time has diminishing returns. The GRU can reliably chain ~5 sequential division steps, but each additional step roughly halves accuracy. This is an exponential decay inherent to the architecture — the hidden state can't maintain precise enough remainder tracking across long chains.

---

## Key Checkpoints (saved, resumable)

| File | Description | Balanced Exact |
|---|---|---|
| `best_3hr_standard.pt` | Best overall model (1-layer GRU h=512) | **56.96%** |
| `best_model_h256.pt` | h=256, trained on random sampling | 97.90% (biased test) |
| `best_1hr_standard.pt` | h=512, 1st hour | 22.77% |
| `best_2hr_standard.pt` | h=512, 2nd hour | 33.83% |

All models: 1-layer GRU, input=33 (1 A-bit + 32 B-bits), output=1 per step, 32 time steps MSB-first.

---

## Summary of What We Learned

1. **Architecture must match the algorithm.** Division is sequential subtract-compare-shift. A seq-to-seq model that feeds all of B at every step (mimicking long division) vastly outperforms flat models.

2. **Curriculum learning is essential** for multi-step algorithmic tasks. Start simple, increase complexity gradually.

3. **CyclicLR + checkpointing** is the most effective training strategy. Cyclic exploration finds good basins; cosine fine-tune squeezes out the last bits of accuracy.

4. **Test set distribution matters enormously.** Random (A, B) sampling gives 97.9% accuracy that looks impressive but hides the fact that the model can't actually divide when A >> B. Balanced sampling reveals the true ~57% accuracy.

5. **The GRU's fundamental limit is the carry chain.** Each quotient bit requires tracking a precise running remainder. The hidden state degrades after ~5 sequential steps, causing exponential accuracy decay. This is not fixable with more data, more capacity, more layers, or more training time.

6. **Bigger is not always better** within a fixed compute budget. h=512 beat h=1024 and h=2048 because larger models need more epochs to converge.

---

## Next Step: Attention

The GRU compresses all context into a fixed hidden state, which degrades over long chains. An **attention mechanism** would let each output step look back at all input positions directly, bypassing the information bottleneck. This is the natural next architectural step.
