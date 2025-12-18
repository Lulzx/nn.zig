const std = @import("std");
const c = @cImport({
    @cInclude("Accelerate/Accelerate.h");
});

// --- Hyperparameters ---
const VOCAB_SIZE: usize = 256;
const D_MODEL: usize = 128; // Per-token embedding
const CONTEXT: usize = 8; // Look at 8 previous tokens
const D_INPUT: usize = D_MODEL * CONTEXT; // 1024 total input dim
const D_HIDDEN: usize = 2048; // Larger hidden layer
const BATCH_SIZE: usize = 128;

const BASE_LR: f32 = 0.001;
const WARMUP_STEPS: usize = 50;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPSILON: f32 = 1e-8;

fn getLR(step: usize) f32 {
    if (step < WARMUP_STEPS) {
        return BASE_LR * @as(f32, @floatFromInt(step + 1)) / @as(f32, WARMUP_STEPS);
    }
    return BASE_LR;
}

// GELU activation (smoother than ReLU, better gradients)
fn gelu(x: []f32) void {
    const sqrt_2_over_pi: f32 = 0.7978845608;
    const coeff: f32 = 0.044715;

    for (x) |*val| {
        const v = val.*;
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const inner = sqrt_2_over_pi * (v + coeff * v * v * v);
        val.* = 0.5 * v * (1.0 + std.math.tanh(inner));
    }
}

fn geluBackward(output: []const f32, input: []const f32, grad: []f32) void {
    const sqrt_2_over_pi: f32 = 0.7978845608;
    const coeff: f32 = 0.044715;

    for (grad, output, input) |*g, _, x| {
        const x3 = x * x * x;
        const inner = sqrt_2_over_pi * (x + coeff * x3);
        const tanh_inner = std.math.tanh(inner);
        const sech2 = 1.0 - tanh_inner * tanh_inner;
        const d_inner = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x);

        // d/dx GELU = 0.5 * (1 + tanh) + 0.5 * x * sech^2 * d_inner
        const d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner;
        g.* *= d_gelu;
    }
}

// Layer Normalization (stabilizes training)
const LayerNorm = struct {
    gamma: []f32,
    beta: []f32,
    dim: usize,
    // Adam state
    m_gamma: []f32,
    v_gamma: []f32,
    m_beta: []f32,
    v_beta: []f32,
    grad_gamma: []f32,
    grad_beta: []f32,
    // Cache for backward
    mean: []f32,
    rstd: []f32,

    pub fn init(alloc: std.mem.Allocator, dim: usize, batch: usize) !LayerNorm {
        const gamma = try alloc.alloc(f32, dim);
        const beta = try alloc.alloc(f32, dim);
        for (gamma) |*g| g.* = 1.0;
        @memset(beta, 0);

        return .{
            .gamma = gamma,
            .beta = beta,
            .dim = dim,
            .m_gamma = try alloc.alloc(f32, dim),
            .v_gamma = try alloc.alloc(f32, dim),
            .m_beta = try alloc.alloc(f32, dim),
            .v_beta = try alloc.alloc(f32, dim),
            .grad_gamma = try alloc.alloc(f32, dim),
            .grad_beta = try alloc.alloc(f32, dim),
            .mean = try alloc.alloc(f32, batch),
            .rstd = try alloc.alloc(f32, batch),
        };
    }

    pub fn forward(self: *LayerNorm, x: []f32, batch: usize) void {
        const eps: f32 = 1e-5;

        for (0..batch) |b| {
            const row = x[b * self.dim ..][0..self.dim];

            // Compute mean
            var sum: f32 = 0;
            for (row) |v| sum += v;
            const mean = sum / @as(f32, @floatFromInt(self.dim));
            self.mean[b] = mean;

            // Compute variance
            var var_sum: f32 = 0;
            for (row) |v| {
                const d = v - mean;
                var_sum += d * d;
            }
            const variance = var_sum / @as(f32, @floatFromInt(self.dim));
            const rstd = 1.0 / @sqrt(variance + eps);
            self.rstd[b] = rstd;

            // Normalize and scale
            for (row, 0..) |*v, i| {
                v.* = (v.* - mean) * rstd * self.gamma[i] + self.beta[i];
            }
        }
    }

    pub fn backward(self: *LayerNorm, x: []const f32, grad: []f32, batch: usize) void {
        const dim_f = @as(f32, @floatFromInt(self.dim));

        for (0..batch) |b| {
            const grad_row = grad[b * self.dim ..][0..self.dim];
            const x_row = x[b * self.dim ..][0..self.dim];
            const mean = self.mean[b];
            const rstd = self.rstd[b];

            // Accumulate gamma/beta gradients
            for (0..self.dim) |i| {
                const x_norm = (x_row[i] - mean) * rstd;
                self.grad_gamma[i] += grad_row[i] * x_norm;
                self.grad_beta[i] += grad_row[i];
            }

            // Compute input gradient
            var sum_grad: f32 = 0;
            var sum_grad_x: f32 = 0;
            for (0..self.dim) |i| {
                sum_grad += grad_row[i] * self.gamma[i];
                sum_grad_x += grad_row[i] * self.gamma[i] * (x_row[i] - mean);
            }

            for (0..self.dim) |i| {
                const x_norm = (x_row[i] - mean) * rstd;
                grad_row[i] = rstd * self.gamma[i] * (grad_row[i] - sum_grad / dim_f - x_norm * sum_grad_x * rstd * rstd / dim_f);
            }
        }
    }

    pub fn applyGradients(self: *LayerNorm, step: usize) void {
        const lr = getLR(step);
        const t = @as(f32, @floatFromInt(step + 1));
        const bc1 = 1.0 - std.math.pow(f32, BETA1, t);
        const bc2 = 1.0 - std.math.pow(f32, BETA2, t);

        for (0..self.dim) |i| {
            // Gamma
            var g = self.grad_gamma[i];
            self.m_gamma[i] = BETA1 * self.m_gamma[i] + (1 - BETA1) * g;
            self.v_gamma[i] = BETA2 * self.v_gamma[i] + (1 - BETA2) * g * g;
            self.gamma[i] -= lr * (self.m_gamma[i] / bc1) / (@sqrt(self.v_gamma[i] / bc2) + EPSILON);

            // Beta
            g = self.grad_beta[i];
            self.m_beta[i] = BETA1 * self.m_beta[i] + (1 - BETA1) * g;
            self.v_beta[i] = BETA2 * self.v_beta[i] + (1 - BETA2) * g * g;
            self.beta[i] -= lr * (self.m_beta[i] / bc1) / (@sqrt(self.v_beta[i] / bc2) + EPSILON);
        }

        @memset(self.grad_gamma, 0);
        @memset(self.grad_beta, 0);
    }
};

pub const Linear = struct {
    weights: []f32,
    bias: []f32,
    in_dim: usize,
    out_dim: usize,
    m_w: []f32,
    v_w: []f32,
    m_b: []f32,
    v_b: []f32,
    grad_w: []f32,
    grad_b: []f32,

    pub fn init(alloc: std.mem.Allocator, in_d: usize, out_d: usize, seed: u64) !Linear {
        const w = try alloc.alloc(f32, out_d * in_d);
        const b = try alloc.alloc(f32, out_d);

        var prng = std.Random.DefaultPrng.init(seed);
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(in_d)));
        for (w) |*val| val.* = prng.random().floatNorm(f32) * scale;
        @memset(b, 0);

        return .{
            .weights = w, .bias = b, .in_dim = in_d, .out_dim = out_d,
            .m_w = try alloc.alloc(f32, out_d * in_d),
            .v_w = try alloc.alloc(f32, out_d * in_d),
            .m_b = try alloc.alloc(f32, out_d),
            .v_b = try alloc.alloc(f32, out_d),
            .grad_w = try alloc.alloc(f32, out_d * in_d),
            .grad_b = try alloc.alloc(f32, out_d),
        };
    }

    pub fn forward(self: *const Linear, input: []const f32, output: []f32, batch: usize) void {
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasTrans,
            @intCast(batch), @intCast(self.out_dim), @intCast(self.in_dim),
            1.0, input.ptr, @intCast(self.in_dim), self.weights.ptr, @intCast(self.in_dim),
            0.0, output.ptr, @intCast(self.out_dim));

        for (0..batch) |b| {
            const row = output[b * self.out_dim ..][0..self.out_dim];
            c.vDSP_vadd(row.ptr, 1, self.bias.ptr, 1, row.ptr, 1, @intCast(self.out_dim));
        }
    }

    pub fn backward(self: *Linear, input: []const f32, out_grad: []const f32, in_grad: []f32, batch: usize) void {
        // Input gradient
        c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans,
            @intCast(batch), @intCast(self.in_dim), @intCast(self.out_dim),
            1.0, out_grad.ptr, @intCast(self.out_dim), self.weights.ptr, @intCast(self.in_dim),
            0.0, in_grad.ptr, @intCast(self.in_dim));

        // Weight gradient
        c.cblas_sgemm(c.CblasRowMajor, c.CblasTrans, c.CblasNoTrans,
            @intCast(self.out_dim), @intCast(self.in_dim), @intCast(batch),
            1.0, out_grad.ptr, @intCast(self.out_dim), input.ptr, @intCast(self.in_dim),
            1.0, self.grad_w.ptr, @intCast(self.in_dim));

        // Bias gradient
        for (0..batch) |b| {
            const row = out_grad[b * self.out_dim ..][0..self.out_dim];
            c.vDSP_vadd(self.grad_b.ptr, 1, row.ptr, 1, self.grad_b.ptr, 1, @intCast(self.out_dim));
        }
    }

    pub fn applyGradients(self: *Linear, step: usize) void {
        const lr = getLR(step);
        const t = @as(f32, @floatFromInt(step + 1));
        const bc1 = 1.0 - std.math.pow(f32, BETA1, t);
        const bc2 = 1.0 - std.math.pow(f32, BETA2, t);

        const wc = self.in_dim * self.out_dim;
        for (0..wc) |i| {
            const g = self.grad_w[i];
            self.m_w[i] = BETA1 * self.m_w[i] + (1 - BETA1) * g;
            self.v_w[i] = BETA2 * self.v_w[i] + (1 - BETA2) * g * g;
            self.weights[i] -= lr * (self.m_w[i] / bc1) / (@sqrt(self.v_w[i] / bc2) + EPSILON);
        }
        for (0..self.out_dim) |i| {
            const g = self.grad_b[i];
            self.m_b[i] = BETA1 * self.m_b[i] + (1 - BETA1) * g;
            self.v_b[i] = BETA2 * self.v_b[i] + (1 - BETA2) * g * g;
            self.bias[i] -= lr * (self.m_b[i] / bc1) / (@sqrt(self.v_b[i] / bc2) + EPSILON);
        }

        @memset(self.grad_w, 0);
        @memset(self.grad_b, 0);
    }
};

pub const Embedding = struct {
    weights: []f32,
    m: []f32,
    v: []f32,
    grad: []f32,
    vocab: usize,
    dim: usize,

    pub fn init(alloc: std.mem.Allocator, vocab: usize, dim: usize) !Embedding {
        const w = try alloc.alloc(f32, vocab * dim);
        var prng = std.Random.DefaultPrng.init(42);
        for (w) |*val| val.* = prng.random().floatNorm(f32) * 0.02;

        const m = try alloc.alloc(f32, vocab * dim);
        const v = try alloc.alloc(f32, vocab * dim);
        const g = try alloc.alloc(f32, vocab * dim);
        @memset(m, 0);
        @memset(v, 0);
        @memset(g, 0);

        return .{ .weights = w, .m = m, .v = v, .grad = g, .vocab = vocab, .dim = dim };
    }

    // Gather CONTEXT tokens into concatenated output
    pub fn gatherContext(self: *const Embedding, data: []const u8, offsets: []const usize, output: []f32, batch: usize) void {
        for (0..batch) |b| {
            const base = offsets[b];
            for (0..CONTEXT) |ctx| {
                const tok = if (base >= CONTEXT - ctx) data[base - (CONTEXT - 1 - ctx)] else 0;
                const src = self.weights[@as(usize, tok) * self.dim ..][0..self.dim];
                const dst = output[(b * CONTEXT + ctx) * self.dim ..][0..self.dim];
                @memcpy(dst, src);
            }
        }
    }

    pub fn accumulateGrad(self: *Embedding, data: []const u8, offsets: []const usize, in_grad: []const f32, batch: usize) void {
        for (0..batch) |b| {
            const base = offsets[b];
            for (0..CONTEXT) |ctx| {
                const tok = if (base >= CONTEXT - ctx) data[base - (CONTEXT - 1 - ctx)] else 0;
                const src = in_grad[(b * CONTEXT + ctx) * self.dim ..][0..self.dim];
                const dst = self.grad[@as(usize, tok) * self.dim ..][0..self.dim];
                c.vDSP_vadd(dst.ptr, 1, src.ptr, 1, dst.ptr, 1, @intCast(self.dim));
            }
        }
    }

    pub fn applyGradients(self: *Embedding, step: usize) void {
        const lr = getLR(step);
        const t = @as(f32, @floatFromInt(step + 1));
        const bc1 = 1.0 - std.math.pow(f32, BETA1, t);
        const bc2 = 1.0 - std.math.pow(f32, BETA2, t);

        for (0..self.vocab * self.dim) |i| {
            const g = self.grad[i];
            self.m[i] = BETA1 * self.m[i] + (1 - BETA1) * g;
            self.v[i] = BETA2 * self.v[i] + (1 - BETA2) * g * g;
            self.weights[i] -= lr * (self.m[i] / bc1) / (@sqrt(self.v[i] / bc2) + EPSILON);
        }
        @memset(self.grad, 0);
    }
};

fn softmaxCE(logits: []f32, targets: []const u8, batch: usize) f32 {
    var loss: f32 = 0;
    for (0..batch) |b| {
        const row = logits[b * VOCAB_SIZE ..][0..VOCAB_SIZE];
        var max_v: f32 = row[0];
        for (row) |l| max_v = @max(max_v, l);
        var sum: f32 = 0;
        for (row) |*l| { l.* = @exp(l.* - max_v); sum += l.*; }
        for (row) |*l| l.* /= sum;
        loss -= @log(@max(row[targets[b]], 1e-10));
    }
    return loss / @as(f32, @floatFromInt(batch));
}

fn softmaxBackward(probs: []f32, targets: []const u8, grad: []f32, batch: usize) void {
    @memcpy(grad, probs);
    const scale = 1.0 / @as(f32, @floatFromInt(batch));
    c.cblas_sscal(@intCast(batch * VOCAB_SIZE), scale, grad.ptr, 1);
    for (0..batch) |b| grad[b * VOCAB_SIZE + targets[b]] -= scale;
}

fn saveModel(path: []const u8, emb: *const Embedding, fc1: *const Linear, fc2: *const Linear) !void {
    const f = try std.fs.cwd().createFile(path, .{});
    defer f.close();
    try f.writeAll(std.mem.sliceAsBytes(emb.weights));
    try f.writeAll(std.mem.sliceAsBytes(fc1.weights));
    try f.writeAll(std.mem.sliceAsBytes(fc1.bias));
    try f.writeAll(std.mem.sliceAsBytes(fc2.weights));
    try f.writeAll(std.mem.sliceAsBytes(fc2.bias));
}

fn loadModel(path: []const u8, emb: *Embedding, fc1: *Linear, fc2: *Linear) !void {
    const f = try std.fs.cwd().openFile(path, .{});
    defer f.close();
    _ = try f.readAll(std.mem.sliceAsBytes(emb.weights));
    _ = try f.readAll(std.mem.sliceAsBytes(fc1.weights));
    _ = try f.readAll(std.mem.sliceAsBytes(fc1.bias));
    _ = try f.readAll(std.mem.sliceAsBytes(fc2.weights));
    _ = try f.readAll(std.mem.sliceAsBytes(fc2.bias));
}

fn generate(emb: *const Embedding, fc1: *const Linear, ln: *LayerNorm, fc2: *const Linear, prompt: []const u8, max_tok: usize, alloc: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const hidden = try alloc.alloc(f32, D_HIDDEN);
    defer alloc.free(hidden);
    const logits = try alloc.alloc(f32, VOCAB_SIZE);
    defer alloc.free(logits);
    const input = try alloc.alloc(f32, D_INPUT);
    defer alloc.free(input);

    // History buffer
    var history: [256]u8 = undefined;
    var hist_len: usize = 0;

    // Copy prompt to history
    for (prompt) |ch| {
        if (hist_len < 256) {
            history[hist_len] = ch;
            hist_len += 1;
        }
    }

    std.debug.print("\n{s}", .{prompt});

    for (0..max_tok) |_| {
        // Build context from history
        for (0..CONTEXT) |ctx| {
            const idx = if (hist_len >= CONTEXT - ctx) hist_len - (CONTEXT - 1 - ctx) - 1 else 0;
            const tok = if (idx < hist_len) history[idx] else 0;
            const src = emb.weights[@as(usize, tok) * D_MODEL ..][0..D_MODEL];
            const dst = input[ctx * D_MODEL ..][0..D_MODEL];
            @memcpy(dst, src);
        }

        fc1.forward(input, hidden, 1);
        ln.forward(hidden, 1);
        gelu(hidden);
        fc2.forward(hidden, logits, 1);

        // Temperature sampling
        var max_v: f32 = logits[0];
        for (logits) |l| max_v = @max(max_v, l);
        var sum: f32 = 0;
        for (logits) |*l| { l.* = @exp((l.* - max_v) / 0.7); sum += l.*; }
        for (logits) |*l| l.* /= sum;

        var r = prng.random().float(f32);
        var next: u8 = 0;
        for (logits, 0..) |p, i| { r -= p; if (r <= 0) { next = @intCast(i); break; } }

        if (next >= 32 and next < 127) std.debug.print("{c}", .{next})
        else if (next == '\n') std.debug.print("\n", .{});

        if (hist_len < 256) { history[hist_len] = next; hist_len += 1; }
    }
    std.debug.print("\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    // Model: Embed(8 tokens) -> FC1 -> LayerNorm -> GELU -> FC2 -> Softmax
    var emb = try Embedding.init(alloc, VOCAB_SIZE, D_MODEL);
    var fc1 = try Linear.init(alloc, D_INPUT, D_HIDDEN, 123);
    var ln = try LayerNorm.init(alloc, D_HIDDEN, BATCH_SIZE);
    var fc2 = try Linear.init(alloc, D_HIDDEN, VOCAB_SIZE, 456);

    if (args.len > 1 and std.mem.eql(u8, args[1], "--generate")) {
        loadModel("model.bin", &emb, &fc1, &fc2) catch {
            std.debug.print("Error: model.bin not found\n", .{});
            return;
        };
        try generate(&emb, &fc1, &ln, &fc2, if (args.len > 2) args[2] else "A: Hello", 200, alloc);
        return;
    }

    const file = std.fs.cwd().openFile("dailydialog.txt", .{}) catch {
        std.debug.print("Error: dailydialog.txt not found\n", .{});
        return;
    };
    defer file.close();
    const size = (try file.stat()).size;
    const data = try std.posix.mmap(null, size, std.posix.PROT.READ, .{ .TYPE = .SHARED }, file.handle, 0);
    defer std.posix.munmap(data);

    std.debug.print("Loaded {d:.1}MB | Context={d} tokens | GELU + LayerNorm\n", .{@as(f32, @floatFromInt(size)) / 1e6, CONTEXT});

    // Buffers
    const offsets = try alloc.alloc(usize, BATCH_SIZE);
    defer alloc.free(offsets);
    const targets = try alloc.alloc(u8, BATCH_SIZE);
    defer alloc.free(targets);
    const input_batch = try alloc.alloc(f32, BATCH_SIZE * D_INPUT);
    defer alloc.free(input_batch);
    const hidden_batch = try alloc.alloc(f32, BATCH_SIZE * D_HIDDEN);
    defer alloc.free(hidden_batch);
    const hidden_pre_gelu = try alloc.alloc(f32, BATCH_SIZE * D_HIDDEN);
    defer alloc.free(hidden_pre_gelu);
    const logits_batch = try alloc.alloc(f32, BATCH_SIZE * VOCAB_SIZE);
    defer alloc.free(logits_batch);
    const hidden_grad = try alloc.alloc(f32, BATCH_SIZE * D_HIDDEN);
    defer alloc.free(hidden_grad);
    const logits_grad = try alloc.alloc(f32, BATCH_SIZE * VOCAB_SIZE);
    defer alloc.free(logits_grad);
    const input_grad = try alloc.alloc(f32, BATCH_SIZE * D_INPUT);
    defer alloc.free(input_grad);

    const start = std.time.milliTimestamp();
    var tokens: usize = 0;
    var loss_sum: f32 = 0;
    var loss_count: usize = 0;

    var step: usize = 0;
    while (step < 1000) : (step += 1) {
        // Sample batch positions (need at least CONTEXT chars before)
        for (0..BATCH_SIZE) |i| {
            const pos = CONTEXT + ((step * BATCH_SIZE + i) * 17) % (size - CONTEXT - 1);
            offsets[i] = pos;
            targets[i] = data[pos];
        }

        // Forward: gather context embeddings
        emb.gatherContext(data, offsets, input_batch, BATCH_SIZE);
        fc1.forward(input_batch, hidden_batch, BATCH_SIZE);
        @memcpy(hidden_pre_gelu, hidden_batch); // Save for backward
        ln.forward(hidden_batch, BATCH_SIZE);
        gelu(hidden_batch);
        fc2.forward(hidden_batch, logits_batch, BATCH_SIZE);

        const loss = softmaxCE(logits_batch, targets, BATCH_SIZE);
        loss_sum += loss;
        loss_count += 1;
        tokens += BATCH_SIZE;

        // Backward
        softmaxBackward(logits_batch, targets, logits_grad, BATCH_SIZE);
        fc2.backward(hidden_batch, logits_grad, hidden_grad, BATCH_SIZE);
        geluBackward(hidden_batch, hidden_pre_gelu, hidden_grad);
        ln.backward(hidden_pre_gelu, hidden_grad, BATCH_SIZE);
        fc1.backward(input_batch, hidden_grad, input_grad, BATCH_SIZE);
        emb.accumulateGrad(data, offsets, input_grad, BATCH_SIZE);

        // Update
        fc1.applyGradients(step);
        ln.applyGradients(step);
        fc2.applyGradients(step);
        emb.applyGradients(step);

        if (step % 50 == 0) {
            const elapsed = @as(f32, @floatFromInt(std.time.milliTimestamp() - start)) / 1000.0;
            const avg = loss_sum / @as(f32, @floatFromInt(loss_count));
            const tps = @as(f32, @floatFromInt(tokens)) / elapsed;
            std.debug.print("\x1b[2J\x1b[HStep: {d} | Loss: {d:.3} | TPS: {d:.0}K | LR: {d:.4}\n", .{step, avg, tps/1000, getLR(step)});
            loss_sum = 0;
            loss_count = 0;
        }
    }

    try saveModel("model.bin", &emb, &fc1, &fc2);
    std.debug.print("\nTraining complete.\n", .{});
}
