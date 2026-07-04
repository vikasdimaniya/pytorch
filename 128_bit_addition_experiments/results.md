# 128-bit Binary Addition Experiments

## Task
Add two 128-bit numbers. Input: two numbers in binary (256 bits total). Output: sum in binary (129 bits). Dataset: 20,000 random train / 5,000 random test pairs.

## Experiment 1: Predict all output bits from final hidden state

Standard approach — the model processes the full input, then predicts all 129 output bits from the last layer or final hidden state.

| Model      | Params   | Bit Acc | Exact Acc | Train Time |
|------------|----------|---------|-----------|------------|
| Linear     | 98,945   | 50.78%  | 0%        | 67s        |
| MLP        | 230,529  | 51.35%  | 0%        | 69s        |
| CNN 1D     | 301,985  | 71.57%  | 0%        | 403s       |
| RNN        | 33,537   | 73.45%  | 0%        | 289s       |
| LSTM       | 84,225   | 75.12%  | 0%        | 682s       |
| GRU        | 67,329   | 73.06%  | 0%        | 616s       |

**Every model fails.** 0% exact match across the board.

## Experiment 2: Seq-to-seq — output one bit per time step

Instead of predicting all 129 bits from the final hidden state, the model outputs one sum bit at each time step. The hidden state only needs to carry the 1-bit carry forward.

| Model      | Params | Bit Acc | Exact Acc | Time to 100% |
|------------|--------|---------|-----------|---------------|
| RNN (h=16) | 337    | 100%    | 100%      | ~18s          |
| GRU (h=16) | 977    | 100%    | 100%      | ~43s          |
| LSTM (h=16)| 1,297  | 100%    | 100%      | ~32s          |

**Every model achieves 100%.** With the right architecture, 128-bit addition is trivial.

## Experiment 3: Minimum model size for 100% accuracy

How small can each model be while still achieving perfect 128-bit addition?

| Model | Min hidden size | Total params | Epochs to 100% |
|-------|-----------------|--------------|-----------------|
| RNN   | h=4             | **37**       | 23              |
| LSTM  | h=3             | 88           | 23              |
| GRU   | h=4             | 101          | 25              |

The RNN achieves perfect 128-bit addition with just **37 parameters**.

## Key Learnings

1. **Architecture matters more than model size.** The first approach used up to 301k params and still got 0% exact accuracy. The seq-to-seq approach used 37 params and got 100%. Throwing more parameters at a badly structured architecture doesn't work.

2. **The bottleneck was information compression, not learning capacity.** In experiment 1, the model had to compress 128 steps of carry history into a single hidden state vector, then decode all 129 bits from it. The hidden state simply couldn't hold enough information. In experiment 2, each output bit is produced at the moment its carry information is available — no compression needed.

3. **Match the architecture to the problem structure.** Addition is inherently sequential (carry propagates left). Seq-to-seq with LSB-first processing mirrors this perfectly: each step is a full adder (2 input bits + carry → 1 sum bit + updated carry). The hidden state learns to be a carry register.

4. **37 parameters encode a full adder.** The RNN with h=4 learns the equivalent of a 1-bit full adder circuit (XOR for sum, AND/OR for carry) and applies it 128 times. A hardware full adder uses ~5 logic gates. The neural network needs h=4 (not h=1) because tanh activation requires a few dimensions to approximate the boolean XOR and carry functions.

5. **Linear/MLP are fundamentally wrong for this at scale.** At 8 bits they could brute-force all carry patterns. At 128 bits the combinatorial explosion makes it impossible — they'd need astronomically more parameters to encode all possible carry chains in a single forward pass.

6. **LSTM's gating helps less than expected here.** The carry in addition is a trivially simple state (0 or 1). LSTM's forget/input/output gates are designed for complex long-range dependencies — overkill for a 1-bit carry. Vanilla RNN with h=4 (37 params) beats LSTM's minimum of h=3 (88 params) on parameter efficiency.
