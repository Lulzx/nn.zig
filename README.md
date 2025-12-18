# nn.zig

A minimal transformer in ~1000 lines of Zig, optimized for Apple Silicon.

## What it does

Trains a character-level language model on dialog data using modern transformer architecture with the Muon optimizer. No frameworks, no dependencies beyond Apple's Accelerate.

```
Step: 0     | Loss: 6.246 | TPS: 20K
Step: 15000 | Loss: 1.581 | TPS: 20K
Step: 30000 | Loss: 1.440 | TPS: 20K
```

## Architecture

```
Input (64 chars) → Embedding → [Layer × 4] → RMSNorm → Linear → Softmax
                                   ↓
                    ┌──────────────────────────────┐
                    │  RMSNorm → Self-Attention    │
                    │      ↓         (RoPE)        │
                    │  + Residual                  │
                    │      ↓                       │
                    │  RMSNorm → SwiGLU FFN        │
                    │      ↓                       │
                    │  + Residual                  │
                    └──────────────────────────────┘
```

Modern LLaMA-style architecture:
- **RMSNorm**: Simpler and faster than LayerNorm
- **RoPE**: Rotary Position Embeddings for position encoding
- **SwiGLU**: Gated activation from LLaMA/PaLM (`SiLU(xW₁) ⊙ xW₂`)
- **Causal Self-Attention**: Single-head with causal masking
- **Muon Optimizer**: Per-feature Newton-like updates with momentum
- **Gradient Clipping**: Global norm clipping for stability

## Muon Optimizer

Muon (by Keller Jordan, 2024) replaces AdamW with a simpler approach:
- Gradient normalized by per-row RMS
- Single momentum buffer (no second moment tracking like Adam)
- Fewer hyperparameters, comparable or better convergence

```
g_norm = g / sqrt(mean(g²) + ε)    # per-row normalization
m = μ * m + g_norm                  # momentum update
w -= lr * (m + λ * w)               # weight update with decay
```

## Performance

| Metric | Value |
|--------|-------|
| Throughput | 20K tokens/sec |
| Final loss | 1.44 @ 30K steps |
| Context | 64 characters |
| Layers | 4 |
| D_MODEL | 256 |

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
const D_MODEL: usize = 256;       // Embedding dimension
const CONTEXT: usize = 64;        // Context window size
const D_FFN: usize = 1024;        // FFN hidden dimension (4x D_MODEL)
const N_LAYERS: usize = 4;        // Number of transformer layers
const BATCH_SIZE: usize = 16;     // Sequences per batch
const BASE_LR: f32 = 0.001;       // Peak learning rate (Muon)
const MUON_MOMENTUM: f32 = 0.95;  // Momentum coefficient
const WEIGHT_DECAY: f32 = 0.01;   // Weight decay
const MAX_STEPS: usize = 30000;   // Training steps
```

## License

MIT
