# nn.zig

A minimal transformer in ~1000 lines of Zig, optimized for Apple Silicon.

## What it does

Trains a character-level language model on dialog data using modern transformer architecture. No frameworks, no dependencies beyond Apple's Accelerate.

```
Step: 0     | Loss: 6.433 | TPS: 64K
Step: 10000 | Loss: 2.131 | TPS: 95K
Step: 20000 | Loss: 2.064 | TPS: 96K
```

## Architecture

```
Input (64 chars) → Embedding → [Layer × 2] → RMSNorm → Linear → Softmax
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
- **AdamW**: With weight decay and cosine LR schedule
- **Gradient Clipping**: Global norm clipping for stability

## Performance

| Metric | Value |
|--------|-------|
| Throughput | 96K tokens/sec |
| Final loss | ~2.0 @ 20K steps |
| Context | 64 characters |
| Layers | 2 |

Speed comes from GEMM batching—processing sequences as matrix multiplications via `cblas_sgemm`.

## Usage

```bash
# Download DailyDialog dataset (~2MB)
python3 fetch_data.py

# Train
zig build run -Doptimize=ReleaseFast

# Generate text (after training)
zig build run -Doptimize=ReleaseFast -- --generate "A: Hello"
```

## Requirements

- macOS with Apple Silicon (uses Accelerate framework)
- Zig 0.13+
- Python 3 (for data download)

## Configuration

Edit constants in `nn.zig`:

```zig
const D_MODEL: usize = 128;      // Embedding dimension
const CONTEXT: usize = 64;       // Context window size
const D_FFN: usize = 512;        // FFN hidden dimension
const N_LAYERS: usize = 2;       // Number of transformer layers
const BATCH_SIZE: usize = 32;    // Sequences per batch
const BASE_LR: f32 = 0.0006;     // Peak learning rate
const MAX_STEPS: usize = 20000;  // Training steps
```

## License

MIT
