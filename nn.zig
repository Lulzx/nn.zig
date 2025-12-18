const std = @import("std");
const c = @cImport({
    @cInclude("Accelerate/Accelerate.h");
});

// --- Modern Transformer Architecture ---
// Based on LLaMA: RMSNorm, RoPE, SwiGLU, Single-head Attention
// Reduced size to prevent overfitting on small dataset (~1.3M tokens)
const VOCAB_SIZE: usize = 256;
const CONTEXT: usize = 64;
const D_MODEL: usize = 128; // Reduced from 256 (better for small data)
const D_HEAD: usize = 128; // Match D_MODEL
const D_FFN: usize = 512; // 4x D_MODEL
const BATCH_SIZE: usize = 32; // Larger batch with smaller model

const BASE_LR: f32 = 0.001; // Muon LR (tuned for this model)
const WARMUP_STEPS: usize = 500;
const MAX_STEPS: usize = 30000;
const GRAD_CLIP: f32 = 1.0; // Gradient clipping
const MUON_MOMENTUM: f32 = 0.95; // Muon momentum
const EPSILON: f32 = 1e-8;
const WEIGHT_DECAY: f32 = 0.01; // Lower weight decay for Muon
const EMA_DECAY: f32 = 0.999; // EMA decay for weight averaging
const REFERENCE_DIM: f32 = 64.0; // Reference dimension for µP scaling
const NS_ITERATIONS: usize = 5; // Newton-Schulz iterations for Muon

// Regularization hyperparameters
const DROPOUT_ATTN: f32 = 0.1; // Dropout after attention
const DROPOUT_FFN: f32 = 0.1; // Dropout after FFN
const LABEL_SMOOTHING: f32 = 0.1; // Label smoothing factor
const VAL_RATIO: f32 = 0.1; // Validation set ratio
const PATIENCE: usize = 5; // Early stopping patience (in 500-step intervals)

// Multi-horizon prediction: predict t+1, t+2, t+3, t+4 from each position
// This extracts 4x more learning signal from the same data
const N_HORIZONS: usize = 4;
const HORIZON_WEIGHTS: [N_HORIZONS]f32 = .{ 1.0, 0.5, 0.25, 0.125 }; // Decay for further horizons

// Cosine LR schedule with warmup
fn getLR(step: usize) f32 {
    if (step < WARMUP_STEPS) {
        return BASE_LR * @as(f32, @floatFromInt(step + 1)) / @as(f32, WARMUP_STEPS);
    }
    const progress = @as(f32, @floatFromInt(step - WARMUP_STEPS)) / @as(f32, @floatFromInt(MAX_STEPS - WARMUP_STEPS));
    return BASE_LR * 0.5 * (1.0 + @cos(std.math.pi * progress));
}

// µP-style LR scaling: different learning rates for different layer types
fn getEmbeddingLR(step: usize) f32 {
    return getLR(step) * (REFERENCE_DIM / @as(f32, @floatFromInt(D_MODEL)));
}

fn getAttentionLR(step: usize) f32 {
    return getLR(step) * @sqrt(REFERENCE_DIM / @as(f32, @floatFromInt(D_MODEL)));
}

fn getOutputLR(step: usize) f32 {
    return getLR(step) * (REFERENCE_DIM / @as(f32, @floatFromInt(D_MODEL)));
}

// Newton-Schulz orthogonalization for true Muon
// Approximates G · (GᵀG)^{-1/2} via iteration: G ← G · (I - 0.5·GᵀG)
fn newtonSchulzOrthogonalize(grad: []f32, dim: usize) void {
    var temp: [D_MODEL * D_MODEL]f32 = undefined;
    var new_grad: [D_MODEL * D_MODEL]f32 = undefined;
    const size = dim * dim;

    // First normalize the gradient to have unit Frobenius norm for stability
    var frob_sq: f32 = 0;
    for (grad[0..size]) |g| frob_sq += g * g;
    const frob = @sqrt(frob_sq + EPSILON);
    for (grad[0..size]) |*g| g.* /= frob;

    for (0..NS_ITERATIONS) |_| {
        // temp = Gᵀ @ G (dim x dim)
        c.cblas_sgemm(c.CblasRowMajor, c.CblasTrans, c.CblasNoTrans,
            @intCast(dim), @intCast(dim), @intCast(dim),
            1.0, grad.ptr, @intCast(dim), grad.ptr, @intCast(dim),
            0.0, &temp, @intCast(dim));

        // temp = I - 0.5 * temp (in-place)
        for (0..dim) |i| {
            for (0..dim) |j| {
                const idx = i * dim + j;
                temp[idx] = (if (i == j) @as(f32, 1.0) else @as(f32, 0.0)) - 0.5 * temp[idx];
            }
        }

        // new_grad = G @ temp
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans,
            @intCast(dim), @intCast(dim), @intCast(dim),
            1.0, grad.ptr, @intCast(dim), &temp, @intCast(dim),
            0.0, &new_grad, @intCast(dim));

        @memcpy(grad[0..size], new_grad[0..size]);
    }

    // Restore scale (optional: can adjust this for different effective learning rates)
    for (grad[0..size]) |*g| g.* *= frob;
}

// Dropout: randomly zero elements during training, scale by 1/(1-p)
fn applyDropout(x: []f32, mask: []f32, p: f32, rand: std.Random) void {
    const scale = 1.0 / (1.0 - p);
    for (x, mask) |*val, *m| {
        if (rand.float(f32) < p) {
            m.* = 0.0;
            val.* = 0.0;
        } else {
            m.* = scale;
            val.* *= scale;
        }
    }
}

fn applyDropoutBackward(grad: []f32, mask: []const f32) void {
    for (grad, mask) |*g, m| g.* *= m;
}

// Adaptive Loss Weighting: Focus gradients on hard predictions, not already-learned patterns
// Tracks per-bigram difficulty using EMA of loss values
const TokenDifficulty = struct {
    ema_loss: []f32, // EMA of loss per bigram (256 * 256 = 65K entries)
    counts: []u32, // Count of observations per bigram

    pub fn init(alloc: std.mem.Allocator) !TokenDifficulty {
        const n = 256 * 256;
        const ema = try alloc.alloc(f32, n);
        const cnt = try alloc.alloc(u32, n);
        @memset(ema, 2.0); // Start assuming everything is moderately hard
        @memset(cnt, 0);
        return .{ .ema_loss = ema, .counts = cnt };
    }

    pub fn deinit(self: *TokenDifficulty, alloc: std.mem.Allocator) void {
        alloc.free(self.ema_loss);
        alloc.free(self.counts);
    }

    pub fn update(self: *TokenDifficulty, prev: u8, curr: u8, loss: f32) void {
        const idx = @as(usize, prev) * 256 + curr;
        const alpha: f32 = 0.01; // Slow update for stability
        self.ema_loss[idx] = (1.0 - alpha) * self.ema_loss[idx] + alpha * loss;
        self.counts[idx] +|= 1; // Saturating add
    }

    pub fn getWeight(self: *const TokenDifficulty, prev: u8, curr: u8) f32 {
        const idx = @as(usize, prev) * 256 + curr;
        // Weight by difficulty: harder patterns get more gradient
        // Floor at 0.1 to never completely ignore anything
        // Cap at 3.0 to prevent instability from rare patterns
        const difficulty = self.ema_loss[idx];
        return @max(0.1, @min(3.0, difficulty / 2.0));
    }

    pub fn getAverageWeight(self: *const TokenDifficulty) f32 {
        var sum: f32 = 0;
        var count: usize = 0;
        for (0..256 * 256) |i| {
            if (self.counts[i] > 0) {
                sum += self.getWeight(@intCast(i / 256), @intCast(i % 256));
                count += 1;
            }
        }
        return if (count > 0) sum / @as(f32, @floatFromInt(count)) else 1.0;
    }
};

// Weighted softmax cross-entropy: weights loss by per-bigram difficulty
fn softmaxCEWeighted(
    logits: []f32,
    tokens: []const u8, // input tokens (for bigram context)
    targets: []const u8,
    n: usize,
    difficulty: *TokenDifficulty,
    training: bool,
    smoothing: f32,
) f32 {
    var loss: f32 = 0;
    var weight_sum: f32 = 0;
    const n_classes = @as(f32, VOCAB_SIZE);

    for (0..n) |i| {
        const row = logits[i * VOCAB_SIZE ..][0..VOCAB_SIZE];

        // Softmax
        var max_v: f32 = row[0];
        for (row) |l| max_v = @max(max_v, l);
        var sum: f32 = 0;
        for (row) |*l| {
            l.* = @exp(l.* - max_v);
            sum += l.*;
        }
        for (row) |*l| l.* /= sum;

        // Compute token loss (with label smoothing)
        const target_log_prob = @log(@max(row[targets[i]], 1e-10));
        var uniform_log_prob: f32 = 0;
        for (row) |p| uniform_log_prob += @log(@max(p, 1e-10));
        uniform_log_prob /= n_classes;
        const token_loss = -((1.0 - smoothing) * target_log_prob + smoothing * uniform_log_prob);

        // Get weight based on how hard this bigram has been
        const prev = if (i > 0) tokens[i - 1] else 0;
        const weight = difficulty.getWeight(prev, targets[i]);

        loss += weight * token_loss;
        weight_sum += weight;

        // Update difficulty tracker during training
        if (training) {
            difficulty.update(prev, targets[i], token_loss);
        }
    }

    return loss / weight_sum;
}

// Weighted backward pass for adaptive loss
fn softmaxBackwardWeighted(
    probs: []f32,
    tokens: []const u8,
    targets: []const u8,
    grad: []f32,
    n: usize,
    difficulty: *const TokenDifficulty,
    smoothing: f32,
) void {
    // Compute total weight for normalization
    var weight_sum: f32 = 0;
    for (0..n) |i| {
        const prev = if (i > 0) tokens[i - 1] else 0;
        weight_sum += difficulty.getWeight(prev, targets[i]);
    }

    const uniform = smoothing / @as(f32, VOCAB_SIZE);

    for (0..n) |i| {
        const prev = if (i > 0) tokens[i - 1] else 0;
        const weight = difficulty.getWeight(prev, targets[i]) / weight_sum;

        const prob_row = probs[i * VOCAB_SIZE ..][0..VOCAB_SIZE];
        const grad_row = grad[i * VOCAB_SIZE ..][0..VOCAB_SIZE];

        for (prob_row, grad_row) |p, *g| {
            g.* = weight * (p - uniform);
        }
        grad_row[targets[i]] -= weight * (1.0 - smoothing);
    }
}

// Label smoothing: (1-ε)·target + ε·uniform
fn softmaxCESmoothed(logits: []f32, targets: []const u8, n: usize, smoothing: f32) f32 {
    var loss: f32 = 0;
    const n_classes = @as(f32, VOCAB_SIZE);

    for (0..n) |i| {
        const row = logits[i * VOCAB_SIZE ..][0..VOCAB_SIZE];
        var max_v: f32 = row[0];
        for (row) |l| max_v = @max(max_v, l);
        var sum: f32 = 0;
        for (row) |*l| {
            l.* = @exp(l.* - max_v);
            sum += l.*;
        }
        for (row) |*l| l.* /= sum;

        // Log probabilities
        const target_log_prob = @log(@max(row[targets[i]], 1e-10));
        var uniform_log_prob: f32 = 0;
        for (row) |p| uniform_log_prob += @log(@max(p, 1e-10));
        uniform_log_prob /= n_classes;

        // Smoothed loss
        loss -= (1.0 - smoothing) * target_log_prob + smoothing * uniform_log_prob;
    }
    return loss / @as(f32, @floatFromInt(n));
}

fn softmaxBackwardSmoothed(probs: []f32, targets: []const u8, grad: []f32, n: usize, smoothing: f32) void {
    const scale = 1.0 / @as(f32, @floatFromInt(n));
    const uniform = smoothing / @as(f32, VOCAB_SIZE);

    for (0..n) |i| {
        const prob_row = probs[i * VOCAB_SIZE ..][0..VOCAB_SIZE];
        const grad_row = grad[i * VOCAB_SIZE ..][0..VOCAB_SIZE];

        for (prob_row, grad_row) |p, *g| {
            // Gradient: p - y_smoothed, where y_smoothed = (1-ε)·one_hot + ε·uniform
            g.* = scale * (p - uniform);
        }
        // Adjust for the one-hot target position
        grad_row[targets[i]] -= scale * (1.0 - smoothing);
    }
}

// RMSNorm (LLaMA) - simpler and faster than LayerNorm
const RMSNorm = struct {
    gamma: []f32,
    ema_gamma: []f32, // EMA weights for generation
    dim: usize,
    // Optimizer state
    m: []f32,
    v: []f32,
    grad: []f32,
    // Cache
    rstd: []f32,

    pub fn init(alloc: std.mem.Allocator, dim: usize, max_batch: usize) !RMSNorm {
        const gamma = try alloc.alloc(f32, dim);
        const ema_gamma = try alloc.alloc(f32, dim);
        for (gamma) |*g| g.* = 1.0;
        for (ema_gamma) |*g| g.* = 1.0;

        return .{
            .gamma = gamma,
            .ema_gamma = ema_gamma,
            .dim = dim,
            .m = try alloc.alloc(f32, dim),
            .v = try alloc.alloc(f32, dim),
            .grad = try alloc.alloc(f32, dim),
            .rstd = try alloc.alloc(f32, max_batch * CONTEXT),
        };
    }

    pub fn forwardEMA(self: *RMSNorm, x: []f32, n: usize) void {
        const eps: f32 = 1e-6;
        for (0..n) |i| {
            const row = x[i * self.dim ..][0..self.dim];
            var ss: f32 = 0;
            for (row) |v| ss += v * v;
            const rstd = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(self.dim)) + eps);
            for (row, 0..) |*v, j| v.* = v.* * rstd * self.ema_gamma[j];
        }
    }

    pub fn forward(self: *RMSNorm, x: []f32, n: usize) void {
        const eps: f32 = 1e-6;
        for (0..n) |i| {
            const row = x[i * self.dim ..][0..self.dim];
            var ss: f32 = 0;
            for (row) |v| ss += v * v;
            const rstd = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(self.dim)) + eps);
            self.rstd[i] = rstd;
            for (row, 0..) |*v, j| v.* = v.* * rstd * self.gamma[j];
        }
    }

    pub fn backward(self: *RMSNorm, x: []const f32, grad: []f32, n: usize) void {
        for (0..n) |i| {
            const grad_row = grad[i * self.dim ..][0..self.dim];
            const x_row = x[i * self.dim ..][0..self.dim];
            const rstd = self.rstd[i];

            // Accumulate gamma gradients
            for (0..self.dim) |j| {
                self.grad[j] += grad_row[j] * x_row[j] * rstd;
            }

            // Input gradient for RMSNorm
            var dot: f32 = 0;
            for (0..self.dim) |j| {
                dot += grad_row[j] * self.gamma[j] * x_row[j];
            }
            const scale = rstd * rstd * rstd / @as(f32, @floatFromInt(self.dim));
            for (0..self.dim) |j| {
                grad_row[j] = rstd * self.gamma[j] * grad_row[j] - scale * dot * x_row[j];
            }
        }
    }

    pub fn applyGradients(self: *RMSNorm, step: usize) void {
        const lr = getLR(step);

        // Muon: normalize gradients by RMS, then apply momentum
        var rms: f32 = 0;
        for (self.grad) |g| rms += g * g;
        rms = @sqrt(rms / @as(f32, @floatFromInt(self.dim)) + EPSILON);

        for (0..self.dim) |i| {
            const g_norm = self.grad[i] / rms;
            self.m[i] = MUON_MOMENTUM * self.m[i] + g_norm;
            self.gamma[i] -= lr * self.m[i];
            // Update EMA
            self.ema_gamma[i] = EMA_DECAY * self.ema_gamma[i] + (1.0 - EMA_DECAY) * self.gamma[i];
        }
        @memset(self.grad, 0);
    }
};

// Single-head Self-Attention with RoPE and QK-Norm
const Attention = struct {
    wq: []f32,
    wk: []f32,
    wv: []f32,
    wo: []f32,
    // EMA weights for generation
    ema_wq: []f32,
    ema_wk: []f32,
    ema_wv: []f32,
    ema_wo: []f32,
    dim: usize,
    // QK-Norm: RMSNorm applied to Q and K before RoPE (stabilizes attention)
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    // Optimizer state for each weight matrix
    m_q: []f32, v_q: []f32, grad_q: []f32,
    m_k: []f32, v_k: []f32, grad_k: []f32,
    m_v: []f32, v_v: []f32, grad_v: []f32,
    m_o: []f32, v_o: []f32, grad_o: []f32,
    // RoPE cache
    cos_cache: []f32,
    sin_cache: []f32,

    pub fn init(alloc: std.mem.Allocator, dim: usize) !Attention {
        const size = dim * dim;
        var prng = std.Random.DefaultPrng.init(789);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));

        const wq = try alloc.alloc(f32, size);
        const wk = try alloc.alloc(f32, size);
        const wv = try alloc.alloc(f32, size);
        const wo = try alloc.alloc(f32, size);
        const ema_wq = try alloc.alloc(f32, size);
        const ema_wk = try alloc.alloc(f32, size);
        const ema_wv = try alloc.alloc(f32, size);
        const ema_wo = try alloc.alloc(f32, size);
        for (wq, ema_wq) |*w, *ew| {
            const val = prng.random().floatNorm(f32) * scale;
            w.* = val;
            ew.* = val;
        }
        for (wk, ema_wk) |*w, *ew| {
            const val = prng.random().floatNorm(f32) * scale;
            w.* = val;
            ew.* = val;
        }
        for (wv, ema_wv) |*w, *ew| {
            const val = prng.random().floatNorm(f32) * scale;
            w.* = val;
            ew.* = val;
        }
        for (wo, ema_wo) |*w, *ew| {
            const val = prng.random().floatNorm(f32) * scale;
            w.* = val;
            ew.* = val;
        }

        // Precompute RoPE frequencies
        const cos_cache = try alloc.alloc(f32, CONTEXT * dim / 2);
        const sin_cache = try alloc.alloc(f32, CONTEXT * dim / 2);
        for (0..CONTEXT) |pos| {
            for (0..dim / 2) |i| {
                const freq = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;
                cos_cache[pos * dim / 2 + i] = @cos(angle);
                sin_cache[pos * dim / 2 + i] = @sin(angle);
            }
        }

        return .{
            .wq = wq, .wk = wk, .wv = wv, .wo = wo,
            .ema_wq = ema_wq, .ema_wk = ema_wk, .ema_wv = ema_wv, .ema_wo = ema_wo,
            .dim = dim,
            .q_norm = try RMSNorm.init(alloc, dim, BATCH_SIZE * CONTEXT),
            .k_norm = try RMSNorm.init(alloc, dim, BATCH_SIZE * CONTEXT),
            .m_q = try alloc.alloc(f32, size), .v_q = try alloc.alloc(f32, size), .grad_q = try alloc.alloc(f32, size),
            .m_k = try alloc.alloc(f32, size), .v_k = try alloc.alloc(f32, size), .grad_k = try alloc.alloc(f32, size),
            .m_v = try alloc.alloc(f32, size), .v_v = try alloc.alloc(f32, size), .grad_v = try alloc.alloc(f32, size),
            .m_o = try alloc.alloc(f32, size), .v_o = try alloc.alloc(f32, size), .grad_o = try alloc.alloc(f32, size),
            .cos_cache = cos_cache, .sin_cache = sin_cache,
        };
    }

    fn applyRoPE(self: *const Attention, x: []f32, seq_len: usize) void {
        for (0..seq_len) |pos| {
            const row = x[pos * self.dim ..][0..self.dim];
            for (0..self.dim / 2) |i| {
                const cos_val = self.cos_cache[pos * self.dim / 2 + i];
                const sin_val = self.sin_cache[pos * self.dim / 2 + i];
                const x0 = row[2 * i];
                const x1 = row[2 * i + 1];
                row[2 * i] = x0 * cos_val - x1 * sin_val;
                row[2 * i + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    fn applyRoPEBackward(self: *const Attention, grad: []f32, seq_len: usize) void {
        for (0..seq_len) |pos| {
            const row = grad[pos * self.dim ..][0..self.dim];
            for (0..self.dim / 2) |i| {
                const cos_val = self.cos_cache[pos * self.dim / 2 + i];
                const sin_val = self.sin_cache[pos * self.dim / 2 + i];
                const g0 = row[2 * i];
                const g1 = row[2 * i + 1];
                // Transpose of rotation matrix
                row[2 * i] = g0 * cos_val + g1 * sin_val;
                row[2 * i + 1] = -g0 * sin_val + g1 * cos_val;
            }
        }
    }

    pub fn forward(self: *Attention, x: []const f32, output: []f32, q: []f32, k: []f32, v: []f32, scores: []f32, seq_len: usize) void {
        const d = self.dim;

        // Q, K, V projections
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(seq_len), @intCast(d), @intCast(d),
            1.0, x.ptr, @intCast(d), self.wq.ptr, @intCast(d), 0.0, q.ptr, @intCast(d));
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(seq_len), @intCast(d), @intCast(d),
            1.0, x.ptr, @intCast(d), self.wk.ptr, @intCast(d), 0.0, k.ptr, @intCast(d));
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(seq_len), @intCast(d), @intCast(d),
            1.0, x.ptr, @intCast(d), self.wv.ptr, @intCast(d), 0.0, v.ptr, @intCast(d));

        // QK-Norm: apply RMSNorm to Q and K before RoPE (stabilizes attention)
        self.q_norm.forward(q, seq_len);
        self.k_norm.forward(k, seq_len);

        // Apply RoPE to Q and K
        self.applyRoPE(q, seq_len);
        self.applyRoPE(k, seq_len);

        // Attention scores: Q @ K^T / sqrt(d)
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(seq_len), @intCast(seq_len), @intCast(d),
            scale, q.ptr, @intCast(d), k.ptr, @intCast(d), 0.0, scores.ptr, @intCast(seq_len));

        // Causal mask + softmax
        for (0..seq_len) |i| {
            const row = scores[i * seq_len ..][0..seq_len];
            // Causal: mask future positions
            for (i + 1..seq_len) |j| row[j] = -1e9;
            // Softmax
            var max_v: f32 = row[0];
            for (row[0..i + 1]) |s| max_v = @max(max_v, s);
            var sum: f32 = 0;
            for (row[0..i + 1]) |*s| { s.* = @exp(s.* - max_v); sum += s.*; }
            for (row[0..i + 1]) |*s| s.* /= sum;
            for (i + 1..seq_len) |j| row[j] = 0;
        }

        // Output: scores @ V
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans,
            @intCast(seq_len), @intCast(d), @intCast(seq_len),
            1.0, scores.ptr, @intCast(seq_len), v.ptr, @intCast(d), 0.0, output.ptr, @intCast(d));

        // Output projection
        const temp = output[0 .. seq_len * d];
        var temp_copy: [CONTEXT * D_HEAD]f32 = undefined;
        @memcpy(temp_copy[0..seq_len * d], temp);
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(seq_len), @intCast(d), @intCast(d),
            1.0, &temp_copy, @intCast(d), self.wo.ptr, @intCast(d), 0.0, output.ptr, @intCast(d));
    }

    // Forward pass using EMA weights (for generation)
    pub fn forwardEMA(self: *Attention, x: []const f32, output: []f32, q: []f32, k: []f32, v: []f32, scores: []f32, seq_len: usize) void {
        const d = self.dim;

        // Q, K, V projections using EMA weights
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(seq_len), @intCast(d), @intCast(d),
            1.0, x.ptr, @intCast(d), self.ema_wq.ptr, @intCast(d), 0.0, q.ptr, @intCast(d));
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(seq_len), @intCast(d), @intCast(d),
            1.0, x.ptr, @intCast(d), self.ema_wk.ptr, @intCast(d), 0.0, k.ptr, @intCast(d));
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(seq_len), @intCast(d), @intCast(d),
            1.0, x.ptr, @intCast(d), self.ema_wv.ptr, @intCast(d), 0.0, v.ptr, @intCast(d));

        // QK-Norm using EMA
        self.q_norm.forwardEMA(q, seq_len);
        self.k_norm.forwardEMA(k, seq_len);

        // Apply RoPE
        self.applyRoPE(q, seq_len);
        self.applyRoPE(k, seq_len);

        // Attention scores
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(seq_len), @intCast(seq_len), @intCast(d),
            scale, q.ptr, @intCast(d), k.ptr, @intCast(d), 0.0, scores.ptr, @intCast(seq_len));

        // Causal mask + softmax
        for (0..seq_len) |i| {
            const row = scores[i * seq_len ..][0..seq_len];
            for (i + 1..seq_len) |j| row[j] = -1e9;
            var max_v: f32 = row[0];
            for (row[0..i + 1]) |s| max_v = @max(max_v, s);
            var sum: f32 = 0;
            for (row[0..i + 1]) |*s| { s.* = @exp(s.* - max_v); sum += s.*; }
            for (row[0..i + 1]) |*s| s.* /= sum;
            for (i + 1..seq_len) |j| row[j] = 0;
        }

        // Output using EMA weights
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans,
            @intCast(seq_len), @intCast(d), @intCast(seq_len),
            1.0, scores.ptr, @intCast(seq_len), v.ptr, @intCast(d), 0.0, output.ptr, @intCast(d));

        var temp_copy: [CONTEXT * D_HEAD]f32 = undefined;
        @memcpy(temp_copy[0..seq_len * d], output[0 .. seq_len * d]);
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(seq_len), @intCast(d), @intCast(d),
            1.0, &temp_copy, @intCast(d), self.ema_wo.ptr, @intCast(d), 0.0, output.ptr, @intCast(d));
    }

    pub fn backward(self: *Attention, x: []const f32, out_grad: []const f32, in_grad: []f32,
                    q: []f32, k: []f32, v: []f32, scores: []const f32, seq_len: usize) void {
        const d = self.dim;

        // Gradient through output projection
        var attn_out_grad: [CONTEXT * D_HEAD]f32 = undefined;
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans,
            @intCast(seq_len), @intCast(d), @intCast(d),
            1.0, out_grad.ptr, @intCast(d), self.wo.ptr, @intCast(d), 0.0, &attn_out_grad, @intCast(d));

        // Accumulate Wo gradient
        var attn_out: [CONTEXT * D_HEAD]f32 = undefined;
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans,
            @intCast(seq_len), @intCast(d), @intCast(seq_len),
            1.0, scores.ptr, @intCast(seq_len), v.ptr, @intCast(d), 0.0, &attn_out, @intCast(d));
        c.cblas_sgemm(c.CblasRowMajor, c.CblasTrans, c.CblasNoTrans,
            @intCast(d), @intCast(d), @intCast(seq_len),
            1.0, out_grad.ptr, @intCast(d), &attn_out, @intCast(d), 1.0, self.grad_o.ptr, @intCast(d));

        // Gradient through attention
        var scores_grad: [CONTEXT * CONTEXT]f32 = undefined;
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(seq_len), @intCast(seq_len), @intCast(d),
            1.0, &attn_out_grad, @intCast(d), v.ptr, @intCast(d), 0.0, &scores_grad, @intCast(seq_len));

        var v_grad: [CONTEXT * D_HEAD]f32 = undefined;
        c.cblas_sgemm(c.CblasRowMajor, c.CblasTrans, c.CblasNoTrans,
            @intCast(seq_len), @intCast(d), @intCast(seq_len),
            1.0, scores.ptr, @intCast(seq_len), &attn_out_grad, @intCast(d), 0.0, &v_grad, @intCast(d));

        // Softmax backward
        for (0..seq_len) |i| {
            const s_row = scores[i * seq_len ..][0..seq_len];
            const g_row = scores_grad[i * seq_len ..][0..seq_len];
            var dot: f32 = 0;
            for (0..i + 1) |j| dot += s_row[j] * g_row[j];
            for (0..i + 1) |j| g_row[j] = s_row[j] * (g_row[j] - dot);
            for (i + 1..seq_len) |j| g_row[j] = 0;
        }

        // Gradient through Q @ K^T
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
        var q_grad: [CONTEXT * D_HEAD]f32 = undefined;
        var k_grad: [CONTEXT * D_HEAD]f32 = undefined;
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans,
            @intCast(seq_len), @intCast(d), @intCast(seq_len),
            scale, &scores_grad, @intCast(seq_len), k.ptr, @intCast(d), 0.0, &q_grad, @intCast(d));
        c.cblas_sgemm(c.CblasRowMajor, c.CblasTrans, c.CblasNoTrans,
            @intCast(seq_len), @intCast(d), @intCast(seq_len),
            scale, &scores_grad, @intCast(seq_len), q.ptr, @intCast(d), 0.0, &k_grad, @intCast(d));

        // RoPE backward
        self.applyRoPEBackward(&q_grad, seq_len);
        self.applyRoPEBackward(&k_grad, seq_len);

        // Accumulate Q, K, V weight gradients
        c.cblas_sgemm(c.CblasRowMajor, c.CblasTrans, c.CblasNoTrans,
            @intCast(d), @intCast(d), @intCast(seq_len),
            1.0, &q_grad, @intCast(d), x.ptr, @intCast(d), 1.0, self.grad_q.ptr, @intCast(d));
        c.cblas_sgemm(c.CblasRowMajor, c.CblasTrans, c.CblasNoTrans,
            @intCast(d), @intCast(d), @intCast(seq_len),
            1.0, &k_grad, @intCast(d), x.ptr, @intCast(d), 1.0, self.grad_k.ptr, @intCast(d));
        c.cblas_sgemm(c.CblasRowMajor, c.CblasTrans, c.CblasNoTrans,
            @intCast(d), @intCast(d), @intCast(seq_len),
            1.0, &v_grad, @intCast(d), x.ptr, @intCast(d), 1.0, self.grad_v.ptr, @intCast(d));

        // Input gradient
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans,
            @intCast(seq_len), @intCast(d), @intCast(d),
            1.0, &q_grad, @intCast(d), self.wq.ptr, @intCast(d), 0.0, in_grad.ptr, @intCast(d));
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans,
            @intCast(seq_len), @intCast(d), @intCast(d),
            1.0, &k_grad, @intCast(d), self.wk.ptr, @intCast(d), 1.0, in_grad.ptr, @intCast(d));
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans,
            @intCast(seq_len), @intCast(d), @intCast(d),
            1.0, &v_grad, @intCast(d), self.wv.ptr, @intCast(d), 1.0, in_grad.ptr, @intCast(d));
    }

    pub fn applyGradients(self: *Attention, step: usize) void {
        // µP-style LR scaling for attention weights
        const lr = getAttentionLR(step);

        // True Muon with Newton-Schulz orthogonalization for square weight matrices
        inline for (.{
            .{ self.wq, self.ema_wq, self.m_q, self.grad_q },
            .{ self.wk, self.ema_wk, self.m_k, self.grad_k },
            .{ self.wv, self.ema_wv, self.m_v, self.grad_v },
            .{ self.wo, self.ema_wo, self.m_o, self.grad_o },
        }) |params| {
            const w = params[0];
            const ema_w = params[1];
            const m = params[2];
            const grad = params[3];

            // Apply Newton-Schulz orthogonalization (true Muon for square matrices)
            newtonSchulzOrthogonalize(grad, self.dim);

            // Apply momentum and weight update
            for (0..self.dim * self.dim) |idx| {
                m[idx] = MUON_MOMENTUM * m[idx] + grad[idx];
                w[idx] -= lr * (m[idx] + WEIGHT_DECAY * w[idx]);
                // Update EMA weights
                ema_w[idx] = EMA_DECAY * ema_w[idx] + (1.0 - EMA_DECAY) * w[idx];
            }
            @memset(grad, 0);
        }

        // Also update QK-Norm parameters
        self.q_norm.applyGradients(step);
        self.k_norm.applyGradients(step);
    }
};

// SwiGLU FFN (LLaMA/PaLM) - better than GELU
const SwiGLU = struct {
    w1: []f32, // gate projection
    w2: []f32, // up projection
    w3: []f32, // down projection
    // EMA weights for generation
    ema_w1: []f32,
    ema_w2: []f32,
    ema_w3: []f32,
    in_dim: usize,
    hidden_dim: usize,
    // Optimizer state
    m1: []f32, v1: []f32, grad1: []f32,
    m2: []f32, v2: []f32, grad2: []f32,
    m3: []f32, v3: []f32, grad3: []f32,
    // Temp buffers for backward
    hidden_buf: []f32,
    gate_grad_buf: []f32,
    up_grad_buf: []f32,

    pub fn init(alloc: std.mem.Allocator, in_d: usize, hidden_d: usize, max_n: usize) !SwiGLU {
        var prng = std.Random.DefaultPrng.init(321);
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(in_d)));

        const w1 = try alloc.alloc(f32, hidden_d * in_d);
        const w2 = try alloc.alloc(f32, hidden_d * in_d);
        const w3 = try alloc.alloc(f32, in_d * hidden_d);
        const ema_w1 = try alloc.alloc(f32, hidden_d * in_d);
        const ema_w2 = try alloc.alloc(f32, hidden_d * in_d);
        const ema_w3 = try alloc.alloc(f32, in_d * hidden_d);
        for (w1, ema_w1) |*w, *ew| {
            const val = prng.random().floatNorm(f32) * scale;
            w.* = val;
            ew.* = val;
        }
        for (w2, ema_w2) |*w, *ew| {
            const val = prng.random().floatNorm(f32) * scale;
            w.* = val;
            ew.* = val;
        }
        for (w3, ema_w3) |*w, *ew| {
            const val = prng.random().floatNorm(f32) * scale;
            w.* = val;
            ew.* = val;
        }

        return .{
            .w1 = w1, .w2 = w2, .w3 = w3,
            .ema_w1 = ema_w1, .ema_w2 = ema_w2, .ema_w3 = ema_w3,
            .in_dim = in_d, .hidden_dim = hidden_d,
            .m1 = try alloc.alloc(f32, hidden_d * in_d),
            .v1 = try alloc.alloc(f32, hidden_d * in_d),
            .grad1 = try alloc.alloc(f32, hidden_d * in_d),
            .m2 = try alloc.alloc(f32, hidden_d * in_d),
            .v2 = try alloc.alloc(f32, hidden_d * in_d),
            .grad2 = try alloc.alloc(f32, hidden_d * in_d),
            .m3 = try alloc.alloc(f32, in_d * hidden_d),
            .v3 = try alloc.alloc(f32, in_d * hidden_d),
            .grad3 = try alloc.alloc(f32, in_d * hidden_d),
            .hidden_buf = try alloc.alloc(f32, max_n * hidden_d),
            .gate_grad_buf = try alloc.alloc(f32, max_n * hidden_d),
            .up_grad_buf = try alloc.alloc(f32, max_n * hidden_d),
        };
    }

    pub fn forward(self: *const SwiGLU, x: []const f32, output: []f32, gate: []f32, up: []f32, n: usize) void {
        // gate = x @ W1, up = x @ W2
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(n), @intCast(self.hidden_dim), @intCast(self.in_dim),
            1.0, x.ptr, @intCast(self.in_dim), self.w1.ptr, @intCast(self.in_dim),
            0.0, gate.ptr, @intCast(self.hidden_dim));
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(n), @intCast(self.hidden_dim), @intCast(self.in_dim),
            1.0, x.ptr, @intCast(self.in_dim), self.w2.ptr, @intCast(self.in_dim),
            0.0, up.ptr, @intCast(self.hidden_dim));

        // SwiGLU: silu(gate) * up
        for (0..n * self.hidden_dim) |i| {
            const g = gate[i];
            const silu = g / (1.0 + @exp(-g)); // SiLU = x * sigmoid(x)
            gate[i] = silu * up[i];
        }

        // output = hidden @ W3
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(n), @intCast(self.in_dim), @intCast(self.hidden_dim),
            1.0, gate.ptr, @intCast(self.hidden_dim), self.w3.ptr, @intCast(self.hidden_dim),
            0.0, output.ptr, @intCast(self.in_dim));
    }

    // Forward pass using EMA weights (for generation)
    pub fn forwardEMA(self: *const SwiGLU, x: []const f32, output: []f32, gate: []f32, up: []f32, n: usize) void {
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(n), @intCast(self.hidden_dim), @intCast(self.in_dim),
            1.0, x.ptr, @intCast(self.in_dim), self.ema_w1.ptr, @intCast(self.in_dim),
            0.0, gate.ptr, @intCast(self.hidden_dim));
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(n), @intCast(self.hidden_dim), @intCast(self.in_dim),
            1.0, x.ptr, @intCast(self.in_dim), self.ema_w2.ptr, @intCast(self.in_dim),
            0.0, up.ptr, @intCast(self.hidden_dim));

        for (0..n * self.hidden_dim) |i| {
            const g = gate[i];
            const silu = g / (1.0 + @exp(-g));
            gate[i] = silu * up[i];
        }

        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(n), @intCast(self.in_dim), @intCast(self.hidden_dim),
            1.0, gate.ptr, @intCast(self.hidden_dim), self.ema_w3.ptr, @intCast(self.hidden_dim),
            0.0, output.ptr, @intCast(self.in_dim));
    }

    pub fn backward(self: *SwiGLU, x: []const f32, out_grad: []const f32, in_grad: []f32,
                    gate_pre: []const f32, up: []const f32, gate: []f32, n: usize) void {
        // Gradient through W3
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans,
            @intCast(n), @intCast(self.hidden_dim), @intCast(self.in_dim),
            1.0, out_grad.ptr, @intCast(self.in_dim), self.w3.ptr, @intCast(self.hidden_dim),
            0.0, gate.ptr, @intCast(self.hidden_dim));

        // Accumulate W3 gradient (need hidden state = silu(gate) * up)
        for (0..n * self.hidden_dim) |i| {
            const g = gate_pre[i];
            const silu = g / (1.0 + @exp(-g));
            self.hidden_buf[i] = silu * up[i];
        }
        c.cblas_sgemm(c.CblasRowMajor, c.CblasTrans, c.CblasNoTrans,
            @intCast(self.in_dim), @intCast(self.hidden_dim), @intCast(n),
            1.0, out_grad.ptr, @intCast(self.in_dim), self.hidden_buf.ptr, @intCast(self.hidden_dim),
            1.0, self.grad3.ptr, @intCast(self.hidden_dim));

        // Gradient through SwiGLU
        for (0..n * self.hidden_dim) |i| {
            const g = gate_pre[i];
            const sig = 1.0 / (1.0 + @exp(-g));
            const silu = g * sig;
            const silu_grad = sig * (1.0 + g * (1.0 - sig));
            self.gate_grad_buf[i] = gate[i] * up[i] * silu_grad;
            self.up_grad_buf[i] = gate[i] * silu;
        }

        // Accumulate W1, W2 gradients
        c.cblas_sgemm(c.CblasRowMajor, c.CblasTrans, c.CblasNoTrans,
            @intCast(self.hidden_dim), @intCast(self.in_dim), @intCast(n),
            1.0, self.gate_grad_buf.ptr, @intCast(self.hidden_dim), x.ptr, @intCast(self.in_dim),
            1.0, self.grad1.ptr, @intCast(self.in_dim));
        c.cblas_sgemm(c.CblasRowMajor, c.CblasTrans, c.CblasNoTrans,
            @intCast(self.hidden_dim), @intCast(self.in_dim), @intCast(n),
            1.0, self.up_grad_buf.ptr, @intCast(self.hidden_dim), x.ptr, @intCast(self.in_dim),
            1.0, self.grad2.ptr, @intCast(self.in_dim));

        // Input gradient
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans,
            @intCast(n), @intCast(self.in_dim), @intCast(self.hidden_dim),
            1.0, self.gate_grad_buf.ptr, @intCast(self.hidden_dim), self.w1.ptr, @intCast(self.in_dim),
            0.0, in_grad.ptr, @intCast(self.in_dim));
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans,
            @intCast(n), @intCast(self.in_dim), @intCast(self.hidden_dim),
            1.0, self.up_grad_buf.ptr, @intCast(self.hidden_dim), self.w2.ptr, @intCast(self.in_dim),
            1.0, in_grad.ptr, @intCast(self.in_dim));
    }

    pub fn applyGradients(self: *SwiGLU, step: usize) void {
        // µP-style LR scaling for FFN weights (same as attention)
        const lr = getAttentionLR(step);

        // Muon for w1, w2 (hidden_dim x in_dim) - per-row normalization
        inline for (.{
            .{ self.w1, self.ema_w1, self.m1, self.grad1 },
            .{ self.w2, self.ema_w2, self.m2, self.grad2 },
        }) |params| {
            const w = params[0];
            const ema_w = params[1];
            const m = params[2];
            const grad = params[3];

            for (0..self.hidden_dim) |row| {
                var rms: f32 = 0;
                for (0..self.in_dim) |col| {
                    const g = grad[row * self.in_dim + col];
                    rms += g * g;
                }
                rms = @sqrt(rms / @as(f32, @floatFromInt(self.in_dim)) + EPSILON);

                for (0..self.in_dim) |col| {
                    const idx = row * self.in_dim + col;
                    const g_norm = grad[idx] / rms;
                    m[idx] = MUON_MOMENTUM * m[idx] + g_norm;
                    w[idx] -= lr * (m[idx] + WEIGHT_DECAY * w[idx]);
                    // Update EMA weights
                    ema_w[idx] = EMA_DECAY * ema_w[idx] + (1.0 - EMA_DECAY) * w[idx];
                }
            }
            @memset(grad, 0);
        }

        // Muon for w3 (in_dim x hidden_dim) - per-row normalization + EMA
        for (0..self.in_dim) |row| {
            var rms: f32 = 0;
            for (0..self.hidden_dim) |col| {
                const g = self.grad3[row * self.hidden_dim + col];
                rms += g * g;
            }
            rms = @sqrt(rms / @as(f32, @floatFromInt(self.hidden_dim)) + EPSILON);

            for (0..self.hidden_dim) |col| {
                const idx = row * self.hidden_dim + col;
                const g_norm = self.grad3[idx] / rms;
                self.m3[idx] = MUON_MOMENTUM * self.m3[idx] + g_norm;
                self.w3[idx] -= lr * (self.m3[idx] + WEIGHT_DECAY * self.w3[idx]);
                // Update EMA weights
                self.ema_w3[idx] = EMA_DECAY * self.ema_w3[idx] + (1.0 - EMA_DECAY) * self.w3[idx];
            }
        }
        @memset(self.grad3, 0);
    }
};

pub const Embedding = struct {
    weights: []f32,
    ema_weights: []f32, // EMA weights for generation
    m: []f32,
    v: []f32,
    grad: []f32,
    vocab: usize,
    dim: usize,

    pub fn init(alloc: std.mem.Allocator, vocab: usize, dim: usize) !Embedding {
        const w = try alloc.alloc(f32, vocab * dim);
        const ema_w = try alloc.alloc(f32, vocab * dim);
        var prng = std.Random.DefaultPrng.init(42);
        for (w, ema_w) |*val, *ema_val| {
            const v = prng.random().floatNorm(f32) * 0.02;
            val.* = v;
            ema_val.* = v;
        }

        const m = try alloc.alloc(f32, vocab * dim);
        const vv = try alloc.alloc(f32, vocab * dim);
        const g = try alloc.alloc(f32, vocab * dim);
        @memset(m, 0);
        @memset(vv, 0);
        @memset(g, 0);

        return .{ .weights = w, .ema_weights = ema_w, .m = m, .v = vv, .grad = g, .vocab = vocab, .dim = dim };
    }

    pub fn gather(self: *const Embedding, tokens: []const u8, output: []f32) void {
        for (tokens, 0..) |tok, i| {
            const src = self.weights[@as(usize, tok) * self.dim ..][0..self.dim];
            const dst = output[i * self.dim ..][0..self.dim];
            @memcpy(dst, src);
        }
    }

    // Gather using EMA weights (for generation)
    pub fn gatherEMA(self: *const Embedding, tokens: []const u8, output: []f32) void {
        for (tokens, 0..) |tok, i| {
            const src = self.ema_weights[@as(usize, tok) * self.dim ..][0..self.dim];
            const dst = output[i * self.dim ..][0..self.dim];
            @memcpy(dst, src);
        }
    }

    pub fn accumulateGrad(self: *Embedding, tokens: []const u8, in_grad: []const f32) void {
        for (tokens, 0..) |tok, i| {
            const src = in_grad[i * self.dim ..][0..self.dim];
            const dst = self.grad[@as(usize, tok) * self.dim ..][0..self.dim];
            c.vDSP_vadd(dst.ptr, 1, src.ptr, 1, dst.ptr, 1, @intCast(self.dim));
        }
    }

    pub fn applyGradients(self: *Embedding, step: usize) void {
        // µP-style LR scaling for embeddings (scale by 1/dim)
        const lr = getEmbeddingLR(step);

        // Muon: normalize per vocabulary token (per row) + EMA update
        for (0..self.vocab) |row| {
            var rms: f32 = 0;
            for (0..self.dim) |col| {
                const g = self.grad[row * self.dim + col];
                rms += g * g;
            }
            rms = @sqrt(rms / @as(f32, @floatFromInt(self.dim)) + EPSILON);

            for (0..self.dim) |col| {
                const idx = row * self.dim + col;
                const g_norm = self.grad[idx] / rms;
                self.m[idx] = MUON_MOMENTUM * self.m[idx] + g_norm;
                self.weights[idx] -= lr * self.m[idx];
                // Update EMA weights
                self.ema_weights[idx] = EMA_DECAY * self.ema_weights[idx] + (1.0 - EMA_DECAY) * self.weights[idx];
            }
        }
        @memset(self.grad, 0);
    }
};

// Output projection (tied with embedding for efficiency)
pub const Linear = struct {
    weights: []f32,
    ema_weights: []f32, // EMA weights for generation
    in_dim: usize,
    out_dim: usize,
    m: []f32,
    v: []f32,
    grad: []f32,

    pub fn init(alloc: std.mem.Allocator, in_d: usize, out_d: usize, seed: u64) !Linear {
        const w = try alloc.alloc(f32, out_d * in_d);
        const ema_w = try alloc.alloc(f32, out_d * in_d);
        var prng = std.Random.DefaultPrng.init(seed);
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(in_d)));
        for (w, ema_w) |*val, *ema_val| {
            const v = prng.random().floatNorm(f32) * scale;
            val.* = v;
            ema_val.* = v;
        }

        return .{
            .weights = w, .ema_weights = ema_w, .in_dim = in_d, .out_dim = out_d,
            .m = try alloc.alloc(f32, out_d * in_d),
            .v = try alloc.alloc(f32, out_d * in_d),
            .grad = try alloc.alloc(f32, out_d * in_d),
        };
    }

    pub fn forward(self: *const Linear, input: []const f32, output: []f32, n: usize) void {
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(n), @intCast(self.out_dim), @intCast(self.in_dim),
            1.0, input.ptr, @intCast(self.in_dim), self.weights.ptr, @intCast(self.in_dim),
            0.0, output.ptr, @intCast(self.out_dim));
    }

    // Forward pass using EMA weights (for generation)
    pub fn forwardEMA(self: *const Linear, input: []const f32, output: []f32, n: usize) void {
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(n), @intCast(self.out_dim), @intCast(self.in_dim),
            1.0, input.ptr, @intCast(self.in_dim), self.ema_weights.ptr, @intCast(self.in_dim),
            0.0, output.ptr, @intCast(self.out_dim));
    }

    pub fn backward(self: *Linear, input: []const f32, out_grad: []const f32, in_grad: []f32, n: usize) void {
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans,
            @intCast(n), @intCast(self.in_dim), @intCast(self.out_dim),
            1.0, out_grad.ptr, @intCast(self.out_dim), self.weights.ptr, @intCast(self.in_dim),
            0.0, in_grad.ptr, @intCast(self.in_dim));

        c.cblas_sgemm(c.CblasRowMajor, c.CblasTrans, c.CblasNoTrans,
            @intCast(self.out_dim), @intCast(self.in_dim), @intCast(n),
            1.0, out_grad.ptr, @intCast(self.out_dim), input.ptr, @intCast(self.in_dim),
            1.0, self.grad.ptr, @intCast(self.in_dim));
    }

    pub fn applyGradients(self: *Linear, step: usize) void {
        // µP-style LR scaling for output projection (scale by 1/dim)
        const lr = getOutputLR(step);

        // Muon: normalize per output neuron (per row) + EMA update
        for (0..self.out_dim) |row| {
            var rms: f32 = 0;
            for (0..self.in_dim) |col| {
                const g = self.grad[row * self.in_dim + col];
                rms += g * g;
            }
            rms = @sqrt(rms / @as(f32, @floatFromInt(self.in_dim)) + EPSILON);

            for (0..self.in_dim) |col| {
                const idx = row * self.in_dim + col;
                const g_norm = self.grad[idx] / rms;
                self.m[idx] = MUON_MOMENTUM * self.m[idx] + g_norm;
                self.weights[idx] -= lr * (self.m[idx] + WEIGHT_DECAY * self.weights[idx]);
                // Update EMA weights
                self.ema_weights[idx] = EMA_DECAY * self.ema_weights[idx] + (1.0 - EMA_DECAY) * self.weights[idx];
            }
        }
        @memset(self.grad, 0);
    }
};

fn softmaxCE(logits: []f32, targets: []const u8, n: usize) f32 {
    var loss: f32 = 0;
    for (0..n) |i| {
        const row = logits[i * VOCAB_SIZE ..][0..VOCAB_SIZE];
        var max_v: f32 = row[0];
        for (row) |l| max_v = @max(max_v, l);
        var sum: f32 = 0;
        for (row) |*l| { l.* = @exp(l.* - max_v); sum += l.*; }
        for (row) |*l| l.* /= sum;
        loss -= @log(@max(row[targets[i]], 1e-10));
    }
    return loss / @as(f32, @floatFromInt(n));
}

fn softmaxBackward(probs: []f32, targets: []const u8, grad: []f32, n: usize) void {
    @memcpy(grad, probs);
    const scale = 1.0 / @as(f32, @floatFromInt(n));
    c.cblas_sscal(@intCast(n * VOCAB_SIZE), scale, grad.ptr, 1);
    for (0..n) |i| grad[i * VOCAB_SIZE + targets[i]] -= scale;
}

const N_LAYERS: usize = 3; // Reduced from 4 (better for small data)

const TransformerLayer = struct {
    attn_norm: RMSNorm,
    attn: Attention,
    ffn_norm: RMSNorm,
    ffn: SwiGLU,
};

const Model = struct {
    emb: Embedding,
    layers: [N_LAYERS]TransformerLayer,
    out_norm: RMSNorm,
    // Multi-horizon prediction heads: each predicts a different future offset
    out_proj: [N_HORIZONS]Linear, // out_proj[0] predicts t+1, out_proj[1] predicts t+2, etc.
};

// Gradient clipping (global norm)
fn gradNormSq(grad: []const f32) f32 {
    var sum: f32 = 0;
    for (grad) |g| sum += g * g;
    return sum;
}

fn scaleGrad(grad: []f32, scale: f32) void {
    for (grad) |*g| g.* *= scale;
}

fn clipGradients(m: *Model) void {
    // Compute global gradient norm
    var norm_sq: f32 = 0;
    norm_sq += gradNormSq(m.emb.grad);
    for (&m.layers) |*layer| {
        norm_sq += gradNormSq(layer.attn_norm.grad);
        norm_sq += gradNormSq(layer.attn.grad_q);
        norm_sq += gradNormSq(layer.attn.grad_k);
        norm_sq += gradNormSq(layer.attn.grad_v);
        norm_sq += gradNormSq(layer.attn.grad_o);
        norm_sq += gradNormSq(layer.ffn_norm.grad);
        norm_sq += gradNormSq(layer.ffn.grad1);
        norm_sq += gradNormSq(layer.ffn.grad2);
        norm_sq += gradNormSq(layer.ffn.grad3);
    }
    norm_sq += gradNormSq(m.out_norm.grad);
    for (&m.out_proj) |*proj| {
        norm_sq += gradNormSq(proj.grad);
    }

    const norm = @sqrt(norm_sq);
    if (norm > GRAD_CLIP) {
        const scale = GRAD_CLIP / norm;
        scaleGrad(m.emb.grad, scale);
        for (&m.layers) |*layer| {
            scaleGrad(layer.attn_norm.grad, scale);
            scaleGrad(layer.attn.grad_q, scale);
            scaleGrad(layer.attn.grad_k, scale);
            scaleGrad(layer.attn.grad_v, scale);
            scaleGrad(layer.attn.grad_o, scale);
            scaleGrad(layer.ffn_norm.grad, scale);
            scaleGrad(layer.ffn.grad1, scale);
            scaleGrad(layer.ffn.grad2, scale);
            scaleGrad(layer.ffn.grad3, scale);
        }
        scaleGrad(m.out_norm.grad, scale);
        for (&m.out_proj) |*proj| {
            scaleGrad(proj.grad, scale);
        }
    }
}

fn saveModel(path: []const u8, m: *const Model) !void {
    const f = try std.fs.cwd().createFile(path, .{});
    defer f.close();
    // Save training weights
    try f.writeAll(std.mem.sliceAsBytes(m.emb.weights));
    for (m.layers) |layer| {
        try f.writeAll(std.mem.sliceAsBytes(layer.attn_norm.gamma));
        try f.writeAll(std.mem.sliceAsBytes(layer.attn.wq));
        try f.writeAll(std.mem.sliceAsBytes(layer.attn.wk));
        try f.writeAll(std.mem.sliceAsBytes(layer.attn.wv));
        try f.writeAll(std.mem.sliceAsBytes(layer.attn.wo));
        try f.writeAll(std.mem.sliceAsBytes(layer.attn.q_norm.gamma));
        try f.writeAll(std.mem.sliceAsBytes(layer.attn.k_norm.gamma));
        try f.writeAll(std.mem.sliceAsBytes(layer.ffn_norm.gamma));
        try f.writeAll(std.mem.sliceAsBytes(layer.ffn.w1));
        try f.writeAll(std.mem.sliceAsBytes(layer.ffn.w2));
        try f.writeAll(std.mem.sliceAsBytes(layer.ffn.w3));
    }
    try f.writeAll(std.mem.sliceAsBytes(m.out_norm.gamma));
    for (m.out_proj) |proj| {
        try f.writeAll(std.mem.sliceAsBytes(proj.weights));
    }
    // Save EMA weights (for generation)
    try f.writeAll(std.mem.sliceAsBytes(m.emb.ema_weights));
    for (m.layers) |layer| {
        try f.writeAll(std.mem.sliceAsBytes(layer.attn_norm.ema_gamma));
        try f.writeAll(std.mem.sliceAsBytes(layer.attn.ema_wq));
        try f.writeAll(std.mem.sliceAsBytes(layer.attn.ema_wk));
        try f.writeAll(std.mem.sliceAsBytes(layer.attn.ema_wv));
        try f.writeAll(std.mem.sliceAsBytes(layer.attn.ema_wo));
        try f.writeAll(std.mem.sliceAsBytes(layer.attn.q_norm.ema_gamma));
        try f.writeAll(std.mem.sliceAsBytes(layer.attn.k_norm.ema_gamma));
        try f.writeAll(std.mem.sliceAsBytes(layer.ffn_norm.ema_gamma));
        try f.writeAll(std.mem.sliceAsBytes(layer.ffn.ema_w1));
        try f.writeAll(std.mem.sliceAsBytes(layer.ffn.ema_w2));
        try f.writeAll(std.mem.sliceAsBytes(layer.ffn.ema_w3));
    }
    try f.writeAll(std.mem.sliceAsBytes(m.out_norm.ema_gamma));
    for (m.out_proj) |proj| {
        try f.writeAll(std.mem.sliceAsBytes(proj.ema_weights));
    }
}

fn loadModel(path: []const u8, m: *Model) !void {
    const f = try std.fs.cwd().openFile(path, .{});
    defer f.close();
    // Load training weights
    _ = try f.readAll(std.mem.sliceAsBytes(m.emb.weights));
    for (&m.layers) |*layer| {
        _ = try f.readAll(std.mem.sliceAsBytes(layer.attn_norm.gamma));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.attn.wq));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.attn.wk));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.attn.wv));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.attn.wo));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.attn.q_norm.gamma));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.attn.k_norm.gamma));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.ffn_norm.gamma));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.ffn.w1));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.ffn.w2));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.ffn.w3));
    }
    _ = try f.readAll(std.mem.sliceAsBytes(m.out_norm.gamma));
    for (&m.out_proj) |*proj| {
        _ = try f.readAll(std.mem.sliceAsBytes(proj.weights));
    }
    // Load EMA weights (for generation)
    _ = try f.readAll(std.mem.sliceAsBytes(m.emb.ema_weights));
    for (&m.layers) |*layer| {
        _ = try f.readAll(std.mem.sliceAsBytes(layer.attn_norm.ema_gamma));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.attn.ema_wq));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.attn.ema_wk));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.attn.ema_wv));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.attn.ema_wo));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.attn.q_norm.ema_gamma));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.attn.k_norm.ema_gamma));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.ffn_norm.ema_gamma));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.ffn.ema_w1));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.ffn.ema_w2));
        _ = try f.readAll(std.mem.sliceAsBytes(layer.ffn.ema_w3));
    }
    _ = try f.readAll(std.mem.sliceAsBytes(m.out_norm.ema_gamma));
    for (&m.out_proj) |*proj| {
        _ = try f.readAll(std.mem.sliceAsBytes(proj.ema_weights));
    }
}

// Quick generation sample for monitoring training progress (uses stack buffers)
fn generateSample(m: *Model, prompt: []const u8, max_tok: usize, rand: std.Random) void {
    var x: [CONTEXT * D_MODEL]f32 = undefined;
    var x_norm: [CONTEXT * D_MODEL]f32 = undefined;
    var attn_out: [CONTEXT * D_MODEL]f32 = undefined;
    var ffn_out: [CONTEXT * D_MODEL]f32 = undefined;
    var q: [CONTEXT * D_HEAD]f32 = undefined;
    var k: [CONTEXT * D_HEAD]f32 = undefined;
    var v: [CONTEXT * D_HEAD]f32 = undefined;
    var scores: [CONTEXT * CONTEXT]f32 = undefined;
    var gate: [CONTEXT * D_FFN]f32 = undefined;
    var up: [CONTEXT * D_FFN]f32 = undefined;
    var logits: [VOCAB_SIZE]f32 = undefined;

    var history: [256]u8 = undefined;
    var hist_len: usize = 0;
    for (prompt) |ch| {
        if (hist_len < 256) { history[hist_len] = ch; hist_len += 1; }
    }

    std.debug.print("  \"{s}", .{prompt});

    for (0..max_tok) |_| {
        const seq_start = if (hist_len > CONTEXT) hist_len - CONTEXT else 0;
        const seq_len = hist_len - seq_start;
        const context = history[seq_start..hist_len];

        m.emb.gatherEMA(context, x[0 .. seq_len * D_MODEL]);

        for (&m.layers) |*layer| {
            @memcpy(x_norm[0 .. seq_len * D_MODEL], x[0 .. seq_len * D_MODEL]);
            layer.attn_norm.forwardEMA(x_norm[0 .. seq_len * D_MODEL], seq_len);
            layer.attn.forwardEMA(x_norm[0 .. seq_len * D_MODEL], attn_out[0 .. seq_len * D_MODEL],
                q[0 .. seq_len * D_HEAD], k[0 .. seq_len * D_HEAD], v[0 .. seq_len * D_HEAD],
                scores[0 .. seq_len * seq_len], seq_len);
            for (0..seq_len * D_MODEL) |i| x[i] += attn_out[i];

            @memcpy(x_norm[0 .. seq_len * D_MODEL], x[0 .. seq_len * D_MODEL]);
            layer.ffn_norm.forwardEMA(x_norm[0 .. seq_len * D_MODEL], seq_len);
            layer.ffn.forwardEMA(x_norm[0 .. seq_len * D_MODEL], ffn_out[0 .. seq_len * D_MODEL],
                gate[0 .. seq_len * D_FFN], up[0 .. seq_len * D_FFN], seq_len);
            for (0..seq_len * D_MODEL) |i| x[i] += ffn_out[i];
        }

        const last_pos = (seq_len - 1) * D_MODEL;
        var last_hidden: [D_MODEL]f32 = undefined;
        @memcpy(&last_hidden, x[last_pos..][0..D_MODEL]);
        m.out_norm.forwardEMA(&last_hidden, 1);
        m.out_proj[0].forwardEMA(&last_hidden, &logits, 1); // Use first head (t+1)

        // Simple temperature + top-k sampling
        const temp: f32 = 0.9;
        var max_v: f32 = logits[0];
        for (&logits) |l| max_v = @max(max_v, l);
        var sum: f32 = 0;
        for (&logits) |*l| { l.* = @exp((l.* - max_v) / temp); sum += l.*; }
        for (&logits) |*l| l.* /= sum;

        var r = rand.float(f32);
        var next: u8 = 0;
        for (logits, 0..) |p, i| { r -= p; if (r <= 0) { next = @intCast(i); break; } }

        if (next >= 32 and next < 127) std.debug.print("{c}", .{next})
        else if (next == '\n') { std.debug.print("\\n", .{}); }

        if (hist_len < 256) { history[hist_len] = next; hist_len += 1; }
    }
    std.debug.print("\"\n", .{});
}

// Generation uses EMA weights for smoother, better quality output
fn generate(m: *Model, prompt: []const u8, max_tok: usize, alloc: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));

    const x = try alloc.alloc(f32, CONTEXT * D_MODEL);
    defer alloc.free(x);
    const attn_out = try alloc.alloc(f32, CONTEXT * D_MODEL);
    defer alloc.free(attn_out);
    const ffn_out = try alloc.alloc(f32, CONTEXT * D_MODEL);
    defer alloc.free(ffn_out);
    const q = try alloc.alloc(f32, CONTEXT * D_HEAD);
    defer alloc.free(q);
    const k = try alloc.alloc(f32, CONTEXT * D_HEAD);
    defer alloc.free(k);
    const v = try alloc.alloc(f32, CONTEXT * D_HEAD);
    defer alloc.free(v);
    const scores = try alloc.alloc(f32, CONTEXT * CONTEXT);
    defer alloc.free(scores);
    const gate = try alloc.alloc(f32, CONTEXT * D_FFN);
    defer alloc.free(gate);
    const up = try alloc.alloc(f32, CONTEXT * D_FFN);
    defer alloc.free(up);
    const logits = try alloc.alloc(f32, VOCAB_SIZE);
    defer alloc.free(logits);

    var history: [512]u8 = undefined;
    var hist_len: usize = 0;
    for (prompt) |ch| {
        if (hist_len < 512) {
            history[hist_len] = ch;
            hist_len += 1;
        }
    }

    std.debug.print("\n{s}", .{prompt});

    for (0..max_tok) |_| {
        const seq_start = if (hist_len > CONTEXT) hist_len - CONTEXT else 0;
        const seq_len = hist_len - seq_start;
        const context = history[seq_start..hist_len];

        // Embedding using EMA weights
        m.emb.gatherEMA(context, x[0 .. seq_len * D_MODEL]);

        // Process through all transformer layers using EMA weights
        var x_norm: [CONTEXT * D_MODEL]f32 = undefined;
        for (&m.layers) |*layer| {
            // Pre-norm + Attention + Residual (using EMA)
            @memcpy(x_norm[0 .. seq_len * D_MODEL], x[0 .. seq_len * D_MODEL]);
            layer.attn_norm.forwardEMA(x_norm[0 .. seq_len * D_MODEL], seq_len);
            layer.attn.forwardEMA(x_norm[0 .. seq_len * D_MODEL], attn_out[0 .. seq_len * D_MODEL],
                           q[0 .. seq_len * D_HEAD], k[0 .. seq_len * D_HEAD], v[0 .. seq_len * D_HEAD],
                           scores[0 .. seq_len * seq_len], seq_len);
            for (0..seq_len * D_MODEL) |i| x[i] += attn_out[i]; // Residual

            // Pre-norm + FFN + Residual (using EMA)
            @memcpy(x_norm[0 .. seq_len * D_MODEL], x[0 .. seq_len * D_MODEL]);
            layer.ffn_norm.forwardEMA(x_norm[0 .. seq_len * D_MODEL], seq_len);
            layer.ffn.forwardEMA(x_norm[0 .. seq_len * D_MODEL], ffn_out[0 .. seq_len * D_MODEL],
                          gate[0 .. seq_len * D_FFN], up[0 .. seq_len * D_FFN], seq_len);
            for (0..seq_len * D_MODEL) |i| x[i] += ffn_out[i]; // Residual
        }

        // Output: last position only (using EMA)
        const last_pos = (seq_len - 1) * D_MODEL;
        var last_hidden: [D_MODEL]f32 = undefined;
        @memcpy(&last_hidden, x[last_pos..][0..D_MODEL]);
        m.out_norm.forwardEMA(&last_hidden, 1);
        m.out_proj[0].forwardEMA(&last_hidden, logits, 1); // Use first head (t+1)

        // Repetition penalty: reduce probability of recently generated tokens
        const rep_penalty: f32 = 1.2;
        const rep_window: usize = 32;
        const window_start = if (hist_len > rep_window) hist_len - rep_window else 0;
        for (history[window_start..hist_len]) |recent| {
            logits[recent] /= rep_penalty;
        }

        // Temperature scaling
        const temp: f32 = 0.8;
        var max_v: f32 = logits[0];
        for (logits) |l| max_v = @max(max_v, l);
        for (logits) |*l| l.* = @exp((l.* - max_v) / temp);

        // Top-k sampling: only consider top k tokens
        const top_k: usize = 40;
        var indices: [VOCAB_SIZE]usize = undefined;
        for (0..VOCAB_SIZE) |i| indices[i] = i;

        // Partial sort to find top-k (simple selection)
        for (0..top_k) |i| {
            var max_idx = i;
            for (i + 1..VOCAB_SIZE) |j| {
                if (logits[indices[j]] > logits[indices[max_idx]]) max_idx = j;
            }
            const tmp = indices[i];
            indices[i] = indices[max_idx];
            indices[max_idx] = tmp;
        }

        // Renormalize over top-k only
        var sum: f32 = 0;
        for (indices[0..top_k]) |idx| sum += logits[idx];

        // Sample from top-k
        var r = prng.random().float(f32) * sum;
        var next: u8 = @intCast(indices[0]);
        for (indices[0..top_k]) |idx| {
            r -= logits[idx];
            if (r <= 0) {
                next = @intCast(idx);
                break;
            }
        }

        if (next >= 32 and next < 127) std.debug.print("{c}", .{next})
        else if (next == '\n') std.debug.print("\n", .{});

        if (hist_len < 512) { history[hist_len] = next; hist_len += 1; }
    }
    std.debug.print("\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    // Initialize model
    var layers: [N_LAYERS]TransformerLayer = undefined;
    for (&layers) |*layer| {
        layer.* = TransformerLayer{
            .attn_norm = try RMSNorm.init(alloc, D_MODEL, BATCH_SIZE * CONTEXT),
            .attn = try Attention.init(alloc, D_HEAD),
            .ffn_norm = try RMSNorm.init(alloc, D_MODEL, BATCH_SIZE * CONTEXT),
            .ffn = try SwiGLU.init(alloc, D_MODEL, D_FFN, BATCH_SIZE * CONTEXT),
        };
    }
    // Initialize multi-horizon prediction heads
    var out_projs: [N_HORIZONS]Linear = undefined;
    for (&out_projs, 0..) |*proj, h| {
        proj.* = try Linear.init(alloc, D_MODEL, VOCAB_SIZE, @intCast(999 + h));
    }
    var model = Model{
        .emb = try Embedding.init(alloc, VOCAB_SIZE, D_MODEL),
        .layers = layers,
        .out_norm = try RMSNorm.init(alloc, D_MODEL, BATCH_SIZE * CONTEXT),
        .out_proj = out_projs,
    };

    if (args.len > 1 and std.mem.eql(u8, args[1], "--generate")) {
        loadModel("model.bin", &model) catch {
            std.debug.print("Error: model.bin not found. Train first.\n", .{});
            return;
        };
        try generate(&model, if (args.len > 2) args[2] else "A: Hello", 200, alloc);
        return;
    }

    const file = std.fs.cwd().openFile("dailydialog.txt", .{}) catch {
        std.debug.print("Error: dailydialog.txt not found. Run: python3 fetch_data.py\n", .{});
        return;
    };
    defer file.close();
    const size = (try file.stat()).size;
    const data = try std.posix.mmap(null, size, std.posix.PROT.READ, .{ .TYPE = .SHARED }, file.handle, 0);
    defer std.posix.munmap(data);

    // Validation split
    const train_size = @as(usize, @intFromFloat(@as(f32, @floatFromInt(size)) * (1.0 - VAL_RATIO)));
    const val_size = size - train_size;

    std.debug.print("Modern Transformer | ctx={d} | d={d} | Multi-Horizon Prediction\n", .{CONTEXT, D_MODEL});
    std.debug.print("Train: {d:.1}MB | Val: {d:.1}MB | {d} steps | horizons={d} (t+1..t+{d})\n",
        .{@as(f32, @floatFromInt(train_size)) / 1e6, @as(f32, @floatFromInt(val_size)) / 1e6,
          MAX_STEPS, N_HORIZONS, N_HORIZONS});

    // Training buffers
    const tokens = try alloc.alloc(u8, BATCH_SIZE * CONTEXT);
    defer alloc.free(tokens);
    const targets = try alloc.alloc(u8, BATCH_SIZE * CONTEXT);
    defer alloc.free(targets);
    const x = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_MODEL);
    defer alloc.free(x);

    // Dropout masks
    var dropout_attn_mask: [N_LAYERS][]f32 = undefined;
    var dropout_ffn_mask: [N_LAYERS][]f32 = undefined;
    for (0..N_LAYERS) |l| {
        dropout_attn_mask[l] = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_MODEL);
        dropout_ffn_mask[l] = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_MODEL);
    }
    defer {
        for (0..N_LAYERS) |l| {
            alloc.free(dropout_attn_mask[l]);
            alloc.free(dropout_ffn_mask[l]);
        }
    }

    // Per-layer intermediate buffers (needed for backward pass)
    // x_orig_*: original input before normalization (for norm backward)
    // x_normed_*: normalized input after norm (for attn/ffn backward)
    var x_orig_attn: [N_LAYERS][]f32 = undefined;
    var x_normed_attn: [N_LAYERS][]f32 = undefined;
    var x_orig_ffn: [N_LAYERS][]f32 = undefined;
    var x_normed_ffn: [N_LAYERS][]f32 = undefined;
    var attn_out: [N_LAYERS][]f32 = undefined;
    var ffn_out: [N_LAYERS][]f32 = undefined;
    var q_buf: [N_LAYERS][]f32 = undefined;
    var k_buf: [N_LAYERS][]f32 = undefined;
    var v_buf: [N_LAYERS][]f32 = undefined;
    var scores_buf: [N_LAYERS][]f32 = undefined;
    var gate_buf: [N_LAYERS][]f32 = undefined;
    var up_buf: [N_LAYERS][]f32 = undefined;
    var gate_pre: [N_LAYERS][]f32 = undefined;

    for (0..N_LAYERS) |l| {
        x_orig_attn[l] = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_MODEL);
        x_normed_attn[l] = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_MODEL);
        x_orig_ffn[l] = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_MODEL);
        x_normed_ffn[l] = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_MODEL);
        attn_out[l] = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_MODEL);
        ffn_out[l] = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_MODEL);
        q_buf[l] = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_HEAD);
        k_buf[l] = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_HEAD);
        v_buf[l] = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_HEAD);
        scores_buf[l] = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * CONTEXT);
        gate_buf[l] = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_FFN);
        up_buf[l] = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_FFN);
        gate_pre[l] = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_FFN);
    }
    defer {
        for (0..N_LAYERS) |l| {
            alloc.free(x_orig_attn[l]);
            alloc.free(x_normed_attn[l]);
            alloc.free(x_orig_ffn[l]);
            alloc.free(x_normed_ffn[l]);
            alloc.free(attn_out[l]);
            alloc.free(ffn_out[l]);
            alloc.free(q_buf[l]);
            alloc.free(k_buf[l]);
            alloc.free(v_buf[l]);
            alloc.free(scores_buf[l]);
            alloc.free(gate_buf[l]);
            alloc.free(up_buf[l]);
            alloc.free(gate_pre[l]);
        }
    }

    const logits = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * VOCAB_SIZE);
    defer alloc.free(logits);
    const logits_grad = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * VOCAB_SIZE);
    defer alloc.free(logits_grad);
    const x_grad = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_MODEL);
    defer alloc.free(x_grad);
    const x_grad_accum = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_MODEL); // Accumulate gradients from all horizons
    defer alloc.free(x_grad_accum);
    // Per-horizon target buffers (targets_h[h][i] = target for position i at horizon h)
    var targets_h: [N_HORIZONS][]u8 = undefined;
    for (0..N_HORIZONS) |h| {
        targets_h[h] = try alloc.alloc(u8, BATCH_SIZE * CONTEXT);
    }
    defer {
        for (0..N_HORIZONS) |h| alloc.free(targets_h[h]);
    }
    // Extra buffers for backward pass
    const x_out = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_MODEL);
    defer alloc.free(x_out);
    const layer_grad = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_MODEL);
    defer alloc.free(layer_grad);
    const layer_in_grad = try alloc.alloc(f32, BATCH_SIZE * CONTEXT * D_MODEL);
    defer alloc.free(layer_in_grad);

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const rand = prng.random();

    const start = std.time.milliTimestamp();
    var total_tokens: usize = 0;
    var loss_sum: f32 = 0;
    var loss_count: usize = 0;

    // Adaptive loss weighting: track per-bigram difficulty
    var difficulty = try TokenDifficulty.init(alloc);
    defer difficulty.deinit(alloc);

    // Early stopping state
    var best_val_loss: f32 = std.math.inf(f32);
    var patience_counter: usize = 0;
    var val_loss: f32 = 0;

    var step: usize = 0;
    while (step < MAX_STEPS) : (step += 1) {
        const seq_len = CONTEXT;

        // Sample random sequences from TRAINING data only
        // Leave room for all horizons (N_HORIZONS extra bytes for targets)
        for (0..BATCH_SIZE) |b| {
            const pos = rand.uintLessThan(usize, train_size - CONTEXT - N_HORIZONS);
            for (0..seq_len) |t| {
                tokens[b * seq_len + t] = data[pos + t];
                targets[b * seq_len + t] = data[pos + t + 1]; // Keep for adaptive weighting
                // Fill multi-horizon targets
                for (0..N_HORIZONS) |h| {
                    targets_h[h][b * seq_len + t] = data[pos + t + h + 1];
                }
            }
        }

        const n = BATCH_SIZE * seq_len;

        // Forward pass
        model.emb.gather(tokens[0..n], x[0 .. n * D_MODEL]);

        // Process through all layers
        for (0..N_LAYERS) |l| {
            // Save original x before attn_norm (for backward)
            @memcpy(x_orig_attn[l][0 .. n * D_MODEL], x[0 .. n * D_MODEL]);
            // Normalize and save for attention
            @memcpy(x_normed_attn[l][0 .. n * D_MODEL], x[0 .. n * D_MODEL]);
            model.layers[l].attn_norm.forward(x_normed_attn[l][0 .. n * D_MODEL], n);

            // Process each sequence separately for causal attention
            for (0..BATCH_SIZE) |b| {
                const offset = b * seq_len;
                model.layers[l].attn.forward(
                    x_normed_attn[l][offset * D_MODEL ..][0 .. seq_len * D_MODEL],
                    attn_out[l][offset * D_MODEL ..][0 .. seq_len * D_MODEL],
                    q_buf[l][offset * D_HEAD ..][0 .. seq_len * D_HEAD],
                    k_buf[l][offset * D_HEAD ..][0 .. seq_len * D_HEAD],
                    v_buf[l][offset * D_HEAD ..][0 .. seq_len * D_HEAD],
                    scores_buf[l][b * seq_len * seq_len ..][0 .. seq_len * seq_len],
                    seq_len
                );
            }
            // Apply dropout to attention output
            applyDropout(attn_out[l][0 .. n * D_MODEL], dropout_attn_mask[l][0 .. n * D_MODEL], DROPOUT_ATTN, rand);
            for (0..n * D_MODEL) |i| x[i] += attn_out[l][i];

            // Save original x before ffn_norm (for backward)
            @memcpy(x_orig_ffn[l][0 .. n * D_MODEL], x[0 .. n * D_MODEL]);
            // Normalize and save for FFN
            @memcpy(x_normed_ffn[l][0 .. n * D_MODEL], x[0 .. n * D_MODEL]);
            model.layers[l].ffn_norm.forward(x_normed_ffn[l][0 .. n * D_MODEL], n);
            model.layers[l].ffn.forward(x_normed_ffn[l][0 .. n * D_MODEL], ffn_out[l][0 .. n * D_MODEL],
                              gate_buf[l][0 .. n * D_FFN], up_buf[l][0 .. n * D_FFN], n);
            // Recompute gate_pre before SiLU for backward
            c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
                @intCast(n), @intCast(D_FFN), @intCast(D_MODEL),
                1.0, x_normed_ffn[l].ptr, @intCast(D_MODEL), model.layers[l].ffn.w1.ptr, @intCast(D_MODEL),
                0.0, gate_pre[l].ptr, @intCast(D_FFN));
            // Apply dropout to FFN output
            applyDropout(ffn_out[l][0 .. n * D_MODEL], dropout_ffn_mask[l][0 .. n * D_MODEL], DROPOUT_FFN, rand);
            for (0..n * D_MODEL) |i| x[i] += ffn_out[l][i];
        }

        // Output norm
        @memcpy(x_out[0 .. n * D_MODEL], x[0 .. n * D_MODEL]);
        model.out_norm.forward(x_out[0 .. n * D_MODEL], n);

        // Multi-horizon prediction: compute loss and gradients for all horizons
        var total_loss: f32 = 0;
        var total_weight: f32 = 0;
        @memset(x_grad_accum[0 .. n * D_MODEL], 0); // Zero accumulator

        for (0..N_HORIZONS) |h| {
            // Forward through horizon head
            model.out_proj[h].forward(x_out[0 .. n * D_MODEL], logits[0 .. n * VOCAB_SIZE], n);

            // Compute weighted loss for this horizon (only use adaptive weighting for h=0)
            const horizon_loss = if (h == 0)
                softmaxCEWeighted(logits[0 .. n * VOCAB_SIZE], tokens[0..n], targets_h[h][0..n], n, &difficulty, true, LABEL_SMOOTHING)
            else
                softmaxCESmoothed(logits[0 .. n * VOCAB_SIZE], targets_h[h][0..n], n, LABEL_SMOOTHING);

            total_loss += HORIZON_WEIGHTS[h] * horizon_loss;
            total_weight += HORIZON_WEIGHTS[h];

            // Backward through this horizon head
            if (h == 0) {
                softmaxBackwardWeighted(logits[0 .. n * VOCAB_SIZE], tokens[0..n], targets_h[h][0..n], logits_grad[0 .. n * VOCAB_SIZE], n, &difficulty, LABEL_SMOOTHING);
            } else {
                softmaxBackwardSmoothed(logits[0 .. n * VOCAB_SIZE], targets_h[h][0..n], logits_grad[0 .. n * VOCAB_SIZE], n, LABEL_SMOOTHING);
            }

            // Backward through projection, accumulate gradient scaled by horizon weight
            model.out_proj[h].backward(x_out[0 .. n * D_MODEL], logits_grad[0 .. n * VOCAB_SIZE], x_grad[0 .. n * D_MODEL], n);
            for (0..n * D_MODEL) |i| {
                x_grad_accum[i] += HORIZON_WEIGHTS[h] * x_grad[i];
            }
        }

        const loss = total_loss / total_weight;
        loss_sum += loss;
        loss_count += 1;
        total_tokens += n;

        // Normalize accumulated gradient and propagate through norm
        for (0..n * D_MODEL) |i| {
            x_grad[i] = x_grad_accum[i] / total_weight;
        }
        model.out_norm.backward(x[0 .. n * D_MODEL], x_grad[0 .. n * D_MODEL], n);

        // Process layers in reverse order
        var l: usize = N_LAYERS;
        while (l > 0) {
            l -= 1;

            // FFN backward (gradient flows through residual)
            @memcpy(layer_grad[0 .. n * D_MODEL], x_grad[0 .. n * D_MODEL]);
            // Apply dropout backward (multiply by same mask)
            applyDropoutBackward(layer_grad[0 .. n * D_MODEL], dropout_ffn_mask[l][0 .. n * D_MODEL]);
            // ffn.backward uses the normalized input (x_normed_ffn)
            model.layers[l].ffn.backward(x_normed_ffn[l][0 .. n * D_MODEL], layer_grad[0 .. n * D_MODEL], layer_in_grad[0 .. n * D_MODEL],
                               gate_pre[l][0 .. n * D_FFN], up_buf[l][0 .. n * D_FFN], gate_buf[l][0 .. n * D_FFN], n);
            // ffn_norm.backward uses the original input before normalization (x_orig_ffn)
            model.layers[l].ffn_norm.backward(x_orig_ffn[l][0 .. n * D_MODEL], layer_in_grad[0 .. n * D_MODEL], n);
            for (0..n * D_MODEL) |i| x_grad[i] += layer_in_grad[i];

            // Attention backward
            @memcpy(layer_grad[0 .. n * D_MODEL], x_grad[0 .. n * D_MODEL]);
            // Apply dropout backward
            applyDropoutBackward(layer_grad[0 .. n * D_MODEL], dropout_attn_mask[l][0 .. n * D_MODEL]);
            for (0..BATCH_SIZE) |b| {
                const offset = b * seq_len;
                // attn.backward uses the normalized input (x_normed_attn)
                model.layers[l].attn.backward(
                    x_normed_attn[l][offset * D_MODEL ..][0 .. seq_len * D_MODEL],
                    layer_grad[offset * D_MODEL ..][0 .. seq_len * D_MODEL],
                    layer_in_grad[offset * D_MODEL ..][0 .. seq_len * D_MODEL],
                    q_buf[l][offset * D_HEAD ..][0 .. seq_len * D_HEAD],
                    k_buf[l][offset * D_HEAD ..][0 .. seq_len * D_HEAD],
                    v_buf[l][offset * D_HEAD ..][0 .. seq_len * D_HEAD],
                    scores_buf[l][b * seq_len * seq_len ..][0 .. seq_len * seq_len],
                    seq_len
                );
            }
            // attn_norm.backward uses the original input before normalization (x_orig_attn)
            model.layers[l].attn_norm.backward(x_orig_attn[l][0 .. n * D_MODEL], layer_in_grad[0 .. n * D_MODEL], n);
            for (0..n * D_MODEL) |i| x_grad[i] += layer_in_grad[i];
        }

        // Embedding backward
        model.emb.accumulateGrad(tokens[0..n], x_grad[0 .. n * D_MODEL]);

        // Clip gradients and update all parameters
        clipGradients(&model);
        model.emb.applyGradients(step);
        for (&model.layers) |*layer| {
            layer.attn_norm.applyGradients(step);
            layer.attn.applyGradients(step);
            layer.ffn_norm.applyGradients(step);
            layer.ffn.applyGradients(step);
        }
        model.out_norm.applyGradients(step);
        for (&model.out_proj) |*proj| {
            proj.applyGradients(step);
        }

        if (step % 100 == 0) {
            const elapsed = @as(f32, @floatFromInt(std.time.milliTimestamp() - start)) / 1000.0;
            const avg_loss = loss_sum / @as(f32, @floatFromInt(loss_count));
            const tps = @as(f32, @floatFromInt(total_tokens)) / elapsed;
            const avg_weight = difficulty.getAverageWeight();
            std.debug.print("\x1b[2J\x1b[HStep: {d}/{d} | Train: {d:.3} | Val: {d:.3} | W: {d:.2} | TPS: {d:.0}K\n",
                          .{step, MAX_STEPS, avg_loss, val_loss, avg_weight, tps/1000});
            loss_sum = 0;
            loss_count = 0;
        }

        // Validation and early stopping every 500 steps
        if (step > 0 and step % 500 == 0) {
            // Compute validation loss (sample from validation set, no dropout)
            var val_loss_sum: f32 = 0;
            const val_batches: usize = 10;
            for (0..val_batches) |_| {
                for (0..BATCH_SIZE) |b| {
                    const val_pos = train_size + rand.uintLessThan(usize, val_size - CONTEXT - 1);
                    for (0..seq_len) |t| {
                        tokens[b * seq_len + t] = data[val_pos + t];
                        targets[b * seq_len + t] = data[val_pos + t + 1];
                    }
                }
                // Forward pass without dropout (use EMA weights for cleaner eval)
                model.emb.gatherEMA(tokens[0..n], x[0 .. n * D_MODEL]);
                for (0..N_LAYERS) |vl| {
                    @memcpy(x_normed_attn[vl][0 .. n * D_MODEL], x[0 .. n * D_MODEL]);
                    model.layers[vl].attn_norm.forwardEMA(x_normed_attn[vl][0 .. n * D_MODEL], n);
                    for (0..BATCH_SIZE) |vb| {
                        const offset = vb * seq_len;
                        model.layers[vl].attn.forwardEMA(
                            x_normed_attn[vl][offset * D_MODEL ..][0 .. seq_len * D_MODEL],
                            attn_out[vl][offset * D_MODEL ..][0 .. seq_len * D_MODEL],
                            q_buf[vl][offset * D_HEAD ..][0 .. seq_len * D_HEAD],
                            k_buf[vl][offset * D_HEAD ..][0 .. seq_len * D_HEAD],
                            v_buf[vl][offset * D_HEAD ..][0 .. seq_len * D_HEAD],
                            scores_buf[vl][vb * seq_len * seq_len ..][0 .. seq_len * seq_len],
                            seq_len
                        );
                    }
                    for (0..n * D_MODEL) |i| x[i] += attn_out[vl][i];
                    @memcpy(x_normed_ffn[vl][0 .. n * D_MODEL], x[0 .. n * D_MODEL]);
                    model.layers[vl].ffn_norm.forwardEMA(x_normed_ffn[vl][0 .. n * D_MODEL], n);
                    model.layers[vl].ffn.forwardEMA(x_normed_ffn[vl][0 .. n * D_MODEL], ffn_out[vl][0 .. n * D_MODEL],
                                      gate_buf[vl][0 .. n * D_FFN], up_buf[vl][0 .. n * D_FFN], n);
                    for (0..n * D_MODEL) |i| x[i] += ffn_out[vl][i];
                }
                @memcpy(x_out[0 .. n * D_MODEL], x[0 .. n * D_MODEL]);
                model.out_norm.forwardEMA(x_out[0 .. n * D_MODEL], n);
                model.out_proj[0].forwardEMA(x_out[0 .. n * D_MODEL], logits[0 .. n * VOCAB_SIZE], n);
                val_loss_sum += softmaxCE(logits[0 .. n * VOCAB_SIZE], targets[0..n], n); // Validate on t+1 only
            }
            val_loss = val_loss_sum / @as(f32, @floatFromInt(val_batches));

            // Show a generation sample to monitor quality
            std.debug.print("Sample:", .{});
            generateSample(&model, "Hello", 32, rand);
            std.debug.print("\n", .{});

            // Early stopping check
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                patience_counter = 0;
                try saveModel("model_best.bin", &model);
            } else {
                patience_counter += 1;
                if (patience_counter >= PATIENCE) {
                    std.debug.print("\nEarly stopping at step {d} (val_loss: {d:.3}, best: {d:.3})\n",
                        .{step, val_loss, best_val_loss});
                    break;
                }
            }
        }
    }

    try saveModel("model.bin", &model);
    std.debug.print("\nTraining complete. Best val_loss: {d:.3}\n", .{best_val_loss});
    std.debug.print("Models saved: model.bin (final), model_best.bin (best)\n", .{});
}
