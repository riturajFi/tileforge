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

## Strict milestone workflowFinal deadline: the linked invoice’s ITC deadline—normally 30 November after that financial year.


- Follow the checklist below strictly, from 00 through 13, top to bottom. Do not skip, reorder, combine, or implement future milestones early.
- Work on one milestone at a time. Each numbered commit adds one working capability.
- Briefly explain the relevant theory, implement the current milestone, and manually verify it by running actual commands. Never run, write, or modify unit tests.
- Run the implementation and inspect numerical results, errors, measurements, and exit statuses. Do not claim verification for commands that were not run.
- Resolve failures before marking a milestone complete. If hardware, credentials, or access block verification, leave it unchecked and record the blocker. Do not advance past it.
- After successful manual verification, update this file's checklist and progress record with what was completed, the commands run, their results, and the next milestone. Include that update in the milestone commit. A commit hash can be recorded afterward when useful; do not invent a hash.
- Before creating any commit, run `git-personal`.
- Commit and push all completed code changes and corresponding documentation, observations, and roadmap updates to `origin/main` before finishing the task. This is standing authorization to push; do not wait for a separate push request. Verify that the remote branch contains the completed commit and report any push blocker. Keep secrets, local environments, and temporary artifacts out of commits, and never force-push over remote work.
- Stop after the requested milestone and report the next step. Proceed to the next milestone when the user requests it.
- Keep this file as the authoritative progress checklist and keep any corresponding README status consistent. Do not optimize an incorrect kernel.

## Commit-by-commit checklist

- [x] **00 — Initialize repository.** Create the README, Python environment, and dependency configuration. Check GPU, PyTorch, and CUDA availability. Deliver a reproducible environment.
- [x] **01 — Implement naive attention.** Write PyTorch QKᵀ → scaling → softmax → V. Start with B=1, H=1, N=4, D=4. Manually verify that the output matches SDPA.
- [x] **02 — Expose the memory bottleneck.** Measure the N × N attention matrix and runtime memory for increasing N. Record baseline measurements.
- [x] **03 — Implement online softmax on CPU.** Process one query against K in small chunks; maintain the running maximum, sum, and numerator accumulator. Verify against full softmax attention.
- [x] **04 — Build CPU tiled attention.** Handle multiple Q rows, K/V tiles, and head dimensions. Never allocate the full attention matrix. Deliver a correct tiled reference.
- [x] **05 — Learn Triton GPU execution.** Implement a small GPU softmax kernel. Practice program IDs, `tl.load`, `tl.store`, masks, and reductions. Deliver a working Triton kernel.
- [x] **06 — Implement tiled QKᵀ in Triton.** Load a Q tile and K tile, perform `tl.dot`, and validate the resulting local scores. Deliver correct tile products.
- [x] **07 — Implement fused Triton attention.** Combine QKᵀ, online softmax, and weighted V accumulation in a single forward kernel. Deliver the first real FlashAttention engine.
- [x] **08 — Add correctness features.** Support causal masking, non-multiple tile sizes, multiple batches/heads, FP16, and BF16. Verify correctness on supported shapes.
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

The user's AMD integrated-GPU laptop can handle the CPU milestones. The Triton GPU kernels, NVIDIA CUDA implementation, and GPU benchmarks require access to a compatible NVIDIA GPU, locally or in the cloud. Milestones 05–06 and 08 were verified on a RunPod NVIDIA A100 80 GB, and milestone 07 on a RunPod NVIDIA RTX 2000 Ada 16 GB, with CUDA-enabled PyTorch and Triton. The local locked environment remains CPU-only, and the verified RunPod images have no `nvcc`; a CUDA compiler is required at milestone 10.

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

### Completed: 02 — Expose the memory bottleneck

- Added `tileforge/memory_baseline.py`, running the existing naive implementation in a fresh Linux subprocess per N, with first-call process RSS measurements, warmed-up timings, SDPA comparisons, and separate operator allocation profiling.
- Ran `UV_CACHE_DIR=/workspace/.cache/uv uv run --locked python -m tileforge.memory_baseline --output observations/2026-10-12/naive-cpu-baseline.json` from `/workspace/tileforge`.
- Exit 0; all seven sizes (128, 256, 512, 1024, 2048, 4096, 8192) matched SDPA. CPU FP32, B=1, H=1, D=64, one compute thread, two warmups, seven timing samples per shape. No unit tests were created, modified, or run.
- At N=8192: 256 MiB per score matrix, 768 MiB across three live matrices, 1031.79 MiB measured process peak RSS, 784.25 MiB first-call high-water increase, 615.508 ms median forward latency.
- Saved raw results and methodology/bottleneck observations under `observations/2026-10-12/`. RSS includes process overhead; these are CPU observations, not GPU measurements or proof of a bandwidth-bound kernel.

### Completed: 03 — Implement online softmax on CPU

- Added `tileforge/online_softmax.py` with `online_attention(q, k, v, chunk_size)` for one CPU FP32 query. Maintains running maximum, normalization sum, and weighted numerator; rescales both accumulators on maximum changes. Forward only, with autograd disabled and O(chunk size + D) scratch space.
- Manually ran `uv run --locked python -m tileforge.online_softmax` with nine configurations: default trace; N=9/D=8/seed=7 with chunks 1, 4, and 32; N=1; N=17/chunk=4 uniform, rising, and negative scenarios; N=8192/D=64/chunk=127/seed=11. Commands ran from `/workspace/tileforge` with `UV_CACHE_DIR=/workspace/.cache/uv`.
- All exited 0, produced finite outputs, and matched full softmax attention and SDPA at `rtol=1e-5, atol=1e-6`. Largest absolute error was `1.19209290e-7` against full softmax and `5.96046448e-8` against SDPA. The uniform run also matched mean(V).
- Saved exact commands, output, chunk traces, and exit codes in `observations/2026-10-12/online-softmax-runs.txt`, with interpretation in `online-softmax.md`. No unit tests were written, changed, or run. No GPU or speedup claims; this milestone computes one query, not the full matrix of queries.

### Completed: 04 — Build CPU tiled attention

- Added `tileforge/tiled.py` with `tiled_attention(q, k, v, query_tile_size, kv_tile_size)` for CPU FP32 inputs `(N, D)`. Keeps separate online-softmax statistics per query row and avoids a full N × N score allocation. Handles partial/oversized tiles, non-contiguous inputs, and N=1; autograd is disabled.
- Ran nine direct CLI configurations, recorded in `observations/2026-10-12/tiled-attention-runs.txt`. All matched naive attention and SDPA at `rtol=1e-5, atol=1e-6`; largest error in those runs was `3.57627869e-7`.
- Extended `tileforge.memory_baseline` with `--implementation tiled`, configurable tiles, and `--skip-profile`. Ran both naive and tiled at N=2048/4096/8192 with D=64, one compute thread, two warmups, seven timed repetitions, and a fresh process per shape. All six outputs matched SDPA. Exact commands and raw JSON are linked from `observations/2026-10-12/tiled-attention.md`.
- At N=8192, peak process RSS was 265.74 MiB tiled versus 1031.77 MiB naive; median CPU latency was 542.815 ms versus 606.434 ms in this run. Process memory includes runtime overhead, and these are not GPU results.
- Ran separate tiled allocation profiling at N=257 with 16/32 tiles: 306 matrix multiplications, matching 17 × 9 tile pairs × 2 products; SDPA maximum error `3.87430191e-7`. The existing default naive profiling path also ran successfully at N=128. No unit tests were written, changed, or run.

### Completed: 05 — Learn Triton GPU execution

- Added `tileforge/triton_softmax.py` with a forward-only CUDA FP32 row softmax. One Triton program handles one row using `tl.program_id`, masked `tl.load`/`tl.store`, maximum and sum reductions, and max-subtracted exponentials.
- Verified actual GPU execution on a RunPod NVIDIA A100 80 GB (capability 8.0), driver 580.159.04, Python 3.12.3, PyTorch 2.8.0+cu128, and Triton 3.4.0. `nvcc` was unavailable, which is acceptable until milestone 10.
- Ran `python3 scripts/check_environment.py --require-cuda` and four CLI configurations: 7×513, 1×1, 3×17, and 128×4096. Every command exited 0, produced finite outputs, and matched `torch.softmax` at `rtol=1e-5, atol=1e-6`. The largest absolute output error was `2.98023224e-8`; the largest row-sum error was `2.38418579e-7`.
- Ran a direct extreme-value check with rows around +10000 and -10000. It exited 0, remained finite, and matched `torch.softmax` with maximum absolute error `5.96046448e-8`.
- Saved exact commands, outputs, and exit codes in `observations/2026-10-12/triton-softmax-runs.txt`, with interpretation and limitations in `triton-softmax.md`. No unit tests were written, changed, or run. No latency, memory, or speedup claims were made.

### Completed: 06 — Implement tiled QKᵀ in Triton

- Added `tileforge/triton_qk.py` with an unscaled CUDA FP32 QKᵀ kernel. A two-dimensional Triton program grid computes 16×16 output tiles, scans the head dimension in 16-wide chunks, uses masked loads with zero padding, accumulates with `tl.dot`, and masks output stores.
- Verified actual GPU execution on a RunPod NVIDIA A100 80 GB (capability 8.0), driver 595.91.07, Python 3.12.3, PyTorch 2.8.0+cu128, and Triton 3.4.0. `nvcc` remained unavailable, which is acceptable until milestone 10.
- Ran `python3 scripts/check_environment.py --require-cuda` and five Q/K shapes: 17×13 by 23×13, 1×1 by 1×1, 32×16 by 48×16, 33×31 by 35×31, and 128×64 by 96×64. Every command exited 0, produced a finite output of the expected shape, and matched `Q @ K.T` at `rtol=1e-5, atol=1e-5`. The largest absolute error was `9.53674316e-6`.
- Saved exact commands, outputs, and exit codes in `observations/2026-10-12/triton-qk-runs.txt`, with interpretation and limitations in `triton-qk.md`. No unit tests were written, changed, or run. No latency, memory, or speedup claims were made.

### Completed: 07 — Implement fused Triton attention

- Added `tileforge/triton_attention.py` with a forward-only, non-causal CUDA FP32 attention kernel. One Triton program owns 16 query rows and scans 16-row K/V tiles while maintaining per-row running maxima, normalization sums, and weighted-V accumulators. It writes only the final output, not full N×N score or weight matrices.
- Kept milestone 08 separate: this first fused kernel accepts one contiguous `(N, D)` sequence, requires N divisible by 16 and D in {16, 32, 64}, and does not support causal masking, batches/heads, FP16, or BF16.
- Verified actual GPU execution on a RunPod NVIDIA RTX 2000 Ada 16 GB (capability 8.9), driver 580.159.04, Python 3.12.3, PyTorch 2.8.0+cu128, and Triton 3.4.0. `nvcc` remained unavailable, which is acceptable until milestone 10.
- Ran `python3 scripts/check_environment.py --require-cuda` and six kernel configurations covering N=16–128, D=16–64, random inputs, uniform scores, and scores whose maxima rise across eight K/V tiles. Every command exited 0, produced finite outputs, and matched naive attention and SDPA at `rtol=1e-4, atol=1e-5`. Largest errors were `4.76837158e-7` against naive attention and `2.02655792e-6` against SDPA.
- Saved exact commands, outputs, and exit codes in `observations/2026-10-12/triton-attention-runs.txt`, with interpretation and limitations in `triton-attention.md`. No unit tests were written, changed, or run. No latency, measured-memory, or speedup claims were made.

### Completed: 08 — Add correctness features

- Extended `tileforge/triton_attention.py` to accept contiguous `(N, D)` or `(B, H, N, D)` Q/K/V tensors, preserving the input shape and dtype. A two-dimensional launch grid maps programs across query tiles and flattened batch/head pairs.
- Added masked Q/K/V loads and output stores for arbitrary positive N and D up to 128, causal score masking, and FP32, FP16, and BF16 paths. Online-softmax statistics and weighted-V accumulation remain FP32; output converts to the input dtype.
- Verified actual GPU execution on a RunPod NVIDIA A100 80 GB (capability 8.0), driver 580.159.04, Python 3.12.3, PyTorch 2.8.0+cu128, and Triton 3.4.0. BF16 support was reported available; `nvcc` remained unavailable, which is acceptable until milestone 10.
- Ran `python3 scripts/check_environment.py --require-cuda` and eight direct kernel checks covering 1×1, preserved 2D input, B=2/H=3, odd N/D tails, causal masking, FP16, BF16, uniform weights, and rising maxima. Every command exited 0 and matched an explicit FP32 reference plus SDPA within dtype-specific tolerances. Largest errors against FP32 reference were `5.96046448e-7` for FP32, `9.54389572e-4` for FP16, and `5.88369370e-3` for BF16.
- Saved exact commands, outputs, and exit codes in `observations/2026-10-12/triton-attention-features-runs.txt`, with interpretation and limits in `triton-attention-features.md`. No unit tests were written, changed, or run. No latency, measured-memory, or speedup claims were made.

### Next: 09 — Optimize the Triton kernel (requires GPU access)

Measure the correct kernel first, then experiment with tile sizes, warp counts, and pipeline stages one variable at a time. Record each performance change without altering correctness scope.
