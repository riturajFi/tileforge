# Agent instructions

## Testing

1. Never run unit tests.
2. Never write unit tests.
3. Never modify unit tests.
4. Test by actually running the code. Manually verify behavior by executing real commands and inspecting their results.

The user provides any required tokens, access, and other credentials. Ask for missing access when it blocks the work; never invent credentials or expose secret values.

## Project: Build Your Own FlashAttention Engine

Build FlashAttention forward-pass kernels from scratch in Triton and CUDA C++, then benchmark both against PyTorch scaled dot-product attention (SDPA).

**Small theory → small commit → verify → next commit.**

The final architecture takes Q, K, and V tensors through either a Triton kernel (Python + GPU) or a CUDA kernel (C++ + GPU). Each fused attention kernel processes Q/K/V tiles, applies online softmax, and writes the output without materializing an N × N attention matrix in GPU high-bandwidth memory (HBM). Compare correctness, latency, and GPU memory against PyTorch SDPA.

## Strict milestone workflow

- Follow the checklist below strictly, from 00 through 13, top to bottom. Do not skip, reorder, combine, or implement future milestones early.
- Work on one milestone at a time. Each numbered commit adds one working capability.
- Briefly explain the relevant theory, implement the current milestone, and manually verify it by running actual commands. Never run, write, or modify unit tests.
- Run the implementation and inspect numerical results, errors, measurements, and exit statuses. Do not claim verification for commands that were not run.
- Resolve failures before marking a milestone complete. If hardware, credentials, or access block verification, leave it unchecked and record the blocker. Do not advance past it.
- After successful manual verification, update this file's checklist and progress record with what was completed, the commands run, their results, and the next milestone. Include that update in the milestone commit. A commit hash can be recorded afterward when useful; do not invent a hash.
- Stop after the requested milestone and report the next step. Proceed to the next milestone when the user requests it.
- Keep this file as the authoritative progress checklist and keep any corresponding README status consistent. Do not optimize an incorrect kernel.

## Commit-by-commit checklist

- [x] **00 — Initialize repository.** Create the README, Python environment, and dependency configuration. Check GPU, PyTorch, and CUDA availability. Deliver a reproducible environment.
- [x] **01 — Implement naive attention.** Write PyTorch QKᵀ → scaling → softmax → V. Start with B=1, H=1, N=4, D=4. Manually verify that the output matches SDPA.
- [ ] **02 — Expose the memory bottleneck.** Measure the N × N attention matrix and runtime memory for increasing N. Record baseline measurements.
- [ ] **03 — Implement online softmax on CPU.** Process one query against K in small chunks; maintain the running maximum, sum, and numerator accumulator. Verify against full softmax attention.
- [ ] **04 — Build CPU tiled attention.** Handle multiple Q rows, K/V tiles, and head dimensions. Never allocate the full attention matrix. Deliver a correct tiled reference.
- [ ] **05 — Learn Triton GPU execution.** Implement a small GPU softmax kernel. Practice program IDs, `tl.load`, `tl.store`, masks, and reductions. Deliver a working Triton kernel.
- [ ] **06 — Implement tiled QKᵀ in Triton.** Load a Q tile and K tile, perform `tl.dot`, and validate the resulting local scores. Deliver correct tile products.
- [ ] **07 — Implement fused Triton attention.** Combine QKᵀ, online softmax, and weighted V accumulation in a single forward kernel. Deliver the first real FlashAttention engine.
- [ ] **08 — Add correctness features.** Support causal masking, non-multiple tile sizes, multiple batches/heads, FP16, and BF16. Verify correctness on supported shapes.
- [ ] **09 — Optimize the Triton kernel.** Experiment with tile sizes, warp counts, and pipeline stages; record performance changes. Deliver a tuned Triton engine.
- [ ] **10 — Build the CUDA C++ foundation.** Create a PyTorch CUDA extension. Learn thread/block mapping, global memory, shared memory, and synchronization. Deliver a callable CUDA kernel.
- [ ] **11 — Implement simple CUDA attention.** Write a correct basic CUDA attention implementation before fusion and tiling. Establish the CUDA numerical baseline.
- [ ] **12 — Implement fused CUDA attention.** Use shared-memory Q/K/V tiles, running softmax statistics, and FP32 accumulators. Optimize only after correctness. Deliver a CUDA FlashAttention forward kernel.
- [ ] **13 — Benchmark and publish.** Compare naive PyTorch, SDPA, Triton, and CUDA across N=128–4096. Plot memory, latency, and speedup. Deliver reproducible benchmarks and a README with results.

## Critical algorithm: commits 03–07

One Triton program processes one Q tile and scans all relevant K/V tiles:

```text
for each Q_tile:
    m = -infinity
    l = 0
    acc = 0
    for each KV_tile:
        scores = Q_tile @ K_tile.T
        scores *= scale

        new_m = max(m, row_max(scores))
        alpha = exp(m - new_m)
        p = exp(scores - new_m)

        l = l * alpha + row_sum(p)
        acc = acc * alpha + p @ V_tile
        m = new_m

    output = acc / l
```

Maintain `m`, `l`, and `acc` per query row and apply broadcasting where needed. Every score tile stays on-chip; no full N × N score matrix gets written to HBM.

## Final benchmark requirements

| Metric | Comparison |
| --- | --- |
| Correctness | Output error against an FP32 reference |
| Latency | Median GPU time in milliseconds |
| GPU memory | Peak allocated memory |
| Speedup | SDPA latency / custom latency |
| Scaling | N = 128, 256, 512, 1024, 2048, 4096 |

Keep batch size, number of heads, head dimension, dtype, and causal mode identical across implementations. Warm up kernels and use CUDA events for timing. Compare against PyTorch's automatic SDPA and, where supported, its explicitly selected FlashAttention backend.

## Hardware and references

The user's AMD integrated-GPU laptop can handle the CPU milestones. The Triton GPU kernels, NVIDIA CUDA implementation, and GPU benchmarks require access to a compatible NVIDIA GPU, locally or in the cloud. The current cloud environment has CPU PyTorch and no usable CUDA device or `nvcc`; prepare and verify the required GPU environment before completing milestone 05. A CUDA compiler is required at milestone 10.

- [Triton — Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Triton — Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton — FlashAttention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)

**Mental model:** Naive attention → N² score matrix in HBM → expensive memory traffic. FlashAttention → Q/K/V tiles → on-chip online softmax → directly write output.

## Progress record

### Completed: 00 — Initialize repository

- Commit: `f52ee08` — `00: initialize reproducible PyTorch development environment`.
- Added the README, `.python-version`, `.gitignore`, `pyproject.toml`, `uv.lock`, and `scripts/check_environment.py`.
- Installed Python 3.12 environment dependencies: PyTorch 2.10.0 CPU and NumPy 2.4.2.
- Manual verification executed from `/workspace/tileforge`:
  - `UV_CACHE_DIR=/workspace/.cache/uv uv sync --locked` — passed, including a repeat run.
  - `UV_CACHE_DIR=/workspace/.cache/uv uv run --locked python scripts/check_environment.py` — passed CPU allocation, matrix multiplication, and SDPA against the known uniform-attention result.
  - `.venv/bin/python scripts/check_environment.py --require-cuda` — returned exit 1 as expected because CUDA is unavailable.
  - `.venv/bin/python scripts/check_environment.py --require-nvcc` — returned exit 1 as expected because `nvcc` is unavailable.
- GPU execution and CUDA compilation remain unverified; they are not requirements for completing the CPU setup milestone.

### Completed: 01 — Implement naive attention

- Added `tileforge/naive.py` with reusable `naive_attention(q, k, v)` implementing QKᵀ → scaling by 1/√D → row softmax → V, plus a runnable comparison with PyTorch SDPA.
- Added README theory, usage, and current milestone status. The implementation uses full attention matrices with no masking or dropout.
- Manual verification executed from `/workspace/tileforge`, with `UV_CACHE_DIR=/workspace/.cache/uv`:
  - `uv run --locked python -m tileforge.naive` — B=1, H=1, N=4, D=4, seed 0; maximum absolute error `2.38418579e-07`.
  - `uv run --locked python -m tileforge.naive --seed 7` — same shape, seed 7; maximum absolute error `1.19209290e-07`.
  - `uv run --locked python -m tileforge.naive --seq-len 9 --head-dim 8` — B=1, H=1, N=9, D=8, seed 0; maximum absolute error `1.19209290e-07`.
- All three commands exited 0 and matched SDPA with `rtol=1e-5, atol=1e-6` on CPU FP32. No unit tests were run, written, or modified. GPU execution was not evaluated.

### Next: 02 — Expose the memory bottleneck

Measure the N × N attention matrix and runtime memory for increasing N, and record baseline measurements. Update this checklist and progress record after successful manual verification.
