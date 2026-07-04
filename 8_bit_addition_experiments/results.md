# 8-bit Binary Addition Experiments

## Task
Add two 8-bit numbers (0–255). Input: two numbers in binary (16 bits total). Output: sum in binary (9 bits). Dataset: all 65,536 possible pairs, 80/20 train/test split.

## Results (50 epochs, hidden=128)

| Model      | Params  | Bit Acc | Exact Acc | Train Time |
|------------|---------|---------|-----------|------------|
| RNN        | 18,057  | 100%    | 100%      | 98s        |
| MLP        | 27,529  | 100%    | 100%      | 57s        |
| CNN 1D     | 35,737  | 100%    | 100%      | 211s       |
| GRU        | 51,849  | 100%    | 100%      | 160s       |
| LSTM       | 68,745  | 100%    | 100%      | 179s       |
| Linear     | 3,337   | 99.66%  | 97.22%    | 47s        |

## Key Learnings

1. **Every architecture except single-layer Linear achieves 100% accuracy.** Binary addition is a fully learnable function. MLP, CNN, RNN, LSTM, GRU all learn it perfectly given enough capacity and training.

2. **Single-layer Linear plateaus at ~97%.** One hidden layer with ReLU can't perfectly model the carry chain across all 8 bit positions. It gets most additions right but fails on edge cases where carries propagate through multiple positions.

3. **RNN is the natural fit for addition.** It hit 100% fastest by epoch (epoch ~10) with the fewest params among the models that reached 100%. Processing LSB-first, the carry flows through the hidden state exactly like hand addition — the architecture matches the problem structure.

4. **MLP works but learns inefficiently.** It has to encode all possible carry patterns implicitly in weight matrices rather than carrying state forward. It needs 27k params and ~40 epochs to reach 100%, compared to RNN's 18k params and ~10 epochs.

5. **All models are overkill for 8-bit addition.** Even 18k params for adding two 8-bit numbers is enormous — a hardware full adder does it with ~40 logic gates. The models have far more capacity than needed, which is why they all succeed. The real test is scaling to larger bit widths.
