# nn.zig

A minimal transformer in ~1700 lines of Zig, optimized for Apple Silicon.

## What it does

Trains a character-level language model on dialog data using modern transformer architecture with adaptive loss weighting. No frameworks, no dependencies beyond Apple's Accelerate.

```
Step: 0     | Train: 6.40 | Val: 0.00 | W: 1.08 | TPS: 10K
Step: 500   | Train: 2.65 | Val: 2.58 | W: 1.32 | TPS: 10K
Sample: "Hello! How are you doing?"
```

## Architecture

```
Input (64 chars) → Embedding → [Layer × 3] → RMSNorm → Linear → Softmax
                                   ↓
                    ┌──────────────────────────────────┐
                    │  RMSNorm → QK-Norm → Attention   │
                    │      ↓         (RoPE)            │
                    │  + Dropout → Residual            │
                    │      ↓                           │
                    │  RMSNorm → SwiGLU FFN            │
                    │      ↓                           │
                    │  + Dropout → Residual            │
                    └──────────────────────────────────┘
```

Modern LLaMA-style architecture:
- **RMSNorm**: Simpler and faster than LayerNorm
- **QK-Norm**: Normalizes Q and K before RoPE for stable attention
- **RoPE**: Rotary Position Embeddings for position encoding
- **SwiGLU**: Gated activation from LLaMA/PaLM (`SiLU(xW₁) ⊙ xW₂`)
- **Causal Self-Attention**: Single-head with causal masking
- **Gradient Clipping**: Global norm clipping for stability

## Adaptive Loss Weighting

A novel training approach that focuses gradients on hard predictions:

**Problem**: Standard training wastes ~80% of gradient signal on already-learned patterns like "q→u" or "the→ ".

**Solution**: Track per-bigram difficulty using EMA of loss values, then weight the loss inversely:

```
difficulty[prev, curr] ← EMA(loss)           # track how hard each bigram is
weight = clamp(difficulty / 2, 0.1, 3.0)     # harder → more gradient
loss = Σ weight[i] · cross_entropy[i]        # focus on what's hard
```

The **W** metric shows average weight over observed bigrams. As training progresses:
- Easy patterns (common bigrams) get downweighted
- W increases as only hard patterns remain
- Model stops "practicing what it knows"

This is implicit curriculum learning without manual scheduling.

## Muon Optimizer

True Muon implementation (Keller Jordan, 2024) with Newton-Schulz orthogonalization:
- Newton-Schulz iterations to orthogonalize gradient matrices
- Single momentum buffer (no second moment tracking like Adam)
- µP-style learning rate scaling per layer type

```
G = newtonSchulz(grad)    # orthogonalize via NS iterations
m = μ * m + G             # momentum update
w -= lr * (m + λ * w)     # weight update with decay
```

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
| Throughput | 10K tokens/sec |
| Context | 64 characters |
| Layers | 3 |
| D_MODEL | 128 |
| Parameters | ~500K |

Speed comes from GEMM batching—processing sequences as matrix multiplications via `cblas_sgemm`.

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
const BASE_LR: f32 = 0.001;       // Peak learning rate
const DROPOUT_ATTN: f32 = 0.1;    // Attention dropout
const DROPOUT_FFN: f32 = 0.1;     // FFN dropout
const LABEL_SMOOTHING: f32 = 0.1; // Label smoothing factor
const VAL_RATIO: f32 = 0.1;       // Validation split ratio
const PATIENCE: usize = 5;        // Early stopping patience
```

## License

MIT
