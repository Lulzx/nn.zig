# nn.zig

A minimal neural network in ~600 lines of Zig, optimized for Apple Silicon.

## What it does

Trains a character-level language model on dialog data. No frameworks, no dependencies beyond Apple's Accelerate.

```
Step: 0   | Loss: 5.872 | TPS: 8K
Step: 500 | Loss: 0.247 | TPS: 10K
Step: 900 | Loss: 0.029 | TPS: 10K
```

## Architecture

```
Input (8 chars) → Embedding → FC1 → LayerNorm → GELU → FC2 → Softmax
     ↓              ↓                                        ↓
  "How are "    [1024-dim]                              P(next char)
```

- **8-token context**: Concatenates embeddings from 8 previous characters
- **GELU activation**: Smoother gradients than ReLU
- **Layer Normalization**: Stabilizes training
- **Adam optimizer**: With linear warmup

## Performance

| Metric | Value |
|--------|-------|
| Throughput | 10K tokens/sec |
| Final loss | ~0.03 @ 1K steps |
| Memory | ~50MB |

Speed comes from GEMM batching—processing 128 tokens as a single matrix multiplication via `cblas_sgemm`.

## Usage

```bash
# Download DailyDialog dataset (~2MB)
python3 fetch_data.py

# Train
zig build run -Doptimize=ReleaseFast
```

## Requirements

- macOS with Apple Silicon (uses Accelerate framework)
- Zig 0.13+
- Python 3 (for data download)

## Configuration

Edit constants in `nn.zig`:

```zig
const D_MODEL: usize = 128;      // Embedding dimension per token
const CONTEXT: usize = 8;        // Context window size
const D_HIDDEN: usize = 2048;    // Hidden layer size
const BATCH_SIZE: usize = 128;   // Tokens per batch
const BASE_LR: f32 = 0.001;      // Learning rate
```

## License

MIT
