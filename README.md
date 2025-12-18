# nn.zig

A minimal transformer in ~1800 lines of Zig, optimized for Apple Silicon.

## What it does

Trains a character-level language model on dialog data using modern transformer architecture with multi-horizon prediction for efficient small-data learning. No frameworks, no dependencies beyond Apple's Accelerate.

```
Step: 0     | Train: 6.56 | Val: 0.00 | W: 1.08 | TPS: 9K
Step: 500   | Train: 2.58 | Val: 2.52 | W: 1.35 | TPS: 9K
Sample: "Hello! How are you doing today?"
```

## Architecture

```
Input (64 chars) → Embedding → [Layer × 3] → RMSNorm → [4 Heads] → Softmax
                                   ↓                        ↓
                    ┌──────────────────────────────┐   ┌─────────────────┐
                    │  RMSNorm → QK-Norm → Attn    │   │ Head 0: t+1     │
                    │      ↓         (RoPE)        │   │ Head 1: t+2     │
                    │  + Dropout → Residual        │   │ Head 2: t+3     │
                    │      ↓                       │   │ Head 3: t+4     │
                    │  RMSNorm → SwiGLU FFN        │   └─────────────────┘
                    │      ↓                       │
                    │  + Dropout → Residual        │
                    └──────────────────────────────┘
```

Modern LLaMA-style architecture:
- **RMSNorm**: Simpler and faster than LayerNorm
- **QK-Norm**: Normalizes Q and K before RoPE for stable attention
- **RoPE**: Rotary Position Embeddings for position encoding
- **SwiGLU**: Gated activation from LLaMA/PaLM (`SiLU(xW₁) ⊙ xW₂`)
- **Causal Self-Attention**: Single-head with causal masking
- **Gradient Clipping**: Global norm clipping for stability

## Multi-Horizon Prediction

**The key innovation for small-data efficiency.**

Standard training: each position predicts only the next byte (t+1).
**Problem**: Wastes information - the sequence contains signal about t+2, t+3, t+4...

**Solution**: Predict multiple future positions from each hidden state:

```
hidden[t] → Head 0 → predict byte[t+1]  (weight 1.0)
         → Head 1 → predict byte[t+2]  (weight 0.5)
         → Head 2 → predict byte[t+3]  (weight 0.25)
         → Head 3 → predict byte[t+4]  (weight 0.125)

total_loss = Σ weight[h] × loss[h]
```

**Benefits**:
- **4x effective dataset size**: Each position provides 4 learning signals
- **Forces abstract representations**: Can't predict t+4 by memorizing local patterns
- **Natural regularization**: Harder to overfit when predicting multiple horizons
- **Implicit curriculum**: t+1 is easy, t+4 is hard - model learns progressively

## Adaptive Loss Weighting

Focuses gradients on hard predictions:

```
difficulty[prev, curr] ← EMA(loss)           # track per-bigram difficulty
weight = clamp(difficulty / 2, 0.1, 3.0)     # harder → more gradient
loss = Σ weight[i] · cross_entropy[i]        # focus on what's hard
```

The **W** metric shows average weight. As training progresses:
- Easy patterns get downweighted
- W increases as model focuses on hard patterns

## Muon Optimizer

True Muon implementation (Keller Jordan, 2024) with Newton-Schulz orthogonalization:
- Newton-Schulz iterations to orthogonalize gradient matrices
- Single momentum buffer (no second moment tracking like Adam)
- µP-style learning rate scaling per layer type

## Regularization

- **Dropout (p=0.1)**: Applied after attention and FFN outputs
- **Label Smoothing (ε=0.1)**: Softens target distribution
- **Validation Split (10%)**: Separate data for monitoring
- **Early Stopping**: Patience-based training termination
- **EMA Weights**: Exponential moving average for smoother generation

## Generation

Improved sampling for coherent text generation:
- **Top-k Sampling (k=40)**: Only sample from top k candidates
- **Temperature Scaling (T=0.8)**: Controls randomness
- **Repetition Penalty (1.2x)**: Reduces repetitive outputs
- **Periodic Samples**: Shows generation quality during training

## Performance

| Metric | Value |
|--------|-------|
| Throughput | 9K tokens/sec |
| Context | 64 characters |
| Layers | 3 |
| D_MODEL | 128 |
| Parameters | ~600K |
| Horizons | 4 (t+1 to t+4) |

## Usage

```bash
# Download DailyDialog dataset (~6.6MB)
python3 fetch_data.py

# Train
zig build run -Doptimize=ReleaseFast

# Generate text (after training)
./zig-out/bin/nn-zig --generate "A: Hello"
```

## Requirements

- macOS with Apple Silicon (uses Accelerate framework)
- Zig 0.13+
- Python 3 (for data download)

## Configuration

Edit constants in `nn.zig`:

```zig
const D_MODEL: usize = 128;       // Embedding dimension
const CONTEXT: usize = 64;        // Context window size
const D_FFN: usize = 512;         // FFN hidden dimension (4x D_MODEL)
const N_LAYERS: usize = 3;        // Number of transformer layers
const BATCH_SIZE: usize = 32;     // Sequences per batch
const N_HORIZONS: usize = 4;      // Multi-horizon prediction (t+1 to t+N)
const HORIZON_WEIGHTS: [4]f32 = .{ 1.0, 0.5, 0.25, 0.125 };
```

## License

MIT
