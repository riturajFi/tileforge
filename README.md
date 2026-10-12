# Tileforge

Build FlashAttention forward-pass kernels from scratch in **Triton** and **CUDA C++**, then compare correctness, latency, and GPU memory with PyTorch scaled dot-product attention (SDPA).

**Small theory → small commit → verify → next commit.**

Progress: commits **00–08 complete**. Next: **09 — optimize the Triton kernel (requires NVIDIA GPU access)**. Follow the authoritative checklist in [AGENTS.md](AGENTS.md).

## Commit 00: reproducible CPU environment

Naive attention materializes an N × N score matrix. FlashAttention instead scans Q/K/V tiles, keeps running softmax statistics and output accumulators, and avoids writing the full score matrix to GPU memory. We will build and verify those pieces incrementally.

Prerequisites: Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.12.23 (supported range: 0.12.x from 0.12.23). From the repository root:

```bash
uv sync --locked
uv run --locked python scripts/check_environment.py
```

`uv sync` creates `.venv`; `uv.lock` pins the resolved dependencies and artifact hashes. `uv run` uses that environment without shell activation. PyTorch comes from its official CPU wheel index; other dependencies come from PyPI. Installation requires access to `download.pytorch.org`, its download redirects, `pypi.org`, and `files.pythonhosted.org`.

The diagnostic reports Python, PyTorch, CUDA runtime support, NVIDIA driver visibility, Triton installation, and `nvcc`. It verifies actual tensor allocation, matrix multiplication, and SDPA against a known uniform-attention result. It always checks CPU execution and checks every CUDA device exposed by PyTorch. An exception or failed assertion returns a nonzero exit status.

Missing CUDA is expected for this CPU environment: milestones 00–04 can run on the AMD laptop or a CPU cloud machine. A CPU PyTorch build cannot use an NVIDIA GPU even if one is attached. `nvidia-smi` reports driver/hardware visibility independently; its absence alone does not prove no GPU exists.

When preparing the later GPU milestones, use a compatible NVIDIA GPU and driver, select an official CUDA-enabled PyTorch build, and update the dependency configuration and lockfile together. Triton is needed at milestone 05; the CUDA toolkit and `nvcc` are needed at milestone 10. PyTorch's bundled CUDA runtime does not supply the CUDA C++ compiler. Then use strict checks:

```bash
uv run --locked python scripts/check_environment.py --require-cuda
uv run --locked python scripts/check_environment.py --require-cuda --require-nvcc
```

These commands deliberately fail in the commit 00 CPU environment. GPU kernel execution and CUDA extension compilation will be validated in their respective milestones.

## Commit 01: naive attention

[`naive_attention`](tileforge/naive.py) implements the four operations explicitly:

1. Compute scores: `Q @ K.transpose(-2, -1)`.
2. Scale by `1 / sqrt(D)` to control the magnitude of dot products.
3. Apply softmax along the key dimension so each query's weights sum to one.
4. Multiply those weights by V to produce the output.

Inputs share shape `(B, H, N, D)`, floating-point dtype, and device. The output has the same shape. This baseline uses no causal mask or dropout and materializes the full `(B, H, N, N)` score matrix. The manual comparison runs on CPU in FP32, starting with `B=1, H=1, N=4, D=4`:

```bash
uv run --locked python -m tileforge.naive
uv run --locked python -m tileforge.naive --seed 7
uv run --locked python -m tileforge.naive --seq-len 9 --head-dim 8
```

Each command runs the implementation and PyTorch SDPA on the same reproducible random inputs, prints both outputs and the maximum absolute error, and exits nonzero if they differ beyond `rtol=1e-5, atol=1e-6`. These are direct execution commands; no unit-test suite is used.

From Python launched in the repository root:

```python
from tileforge.naive import naive_attention

output = naive_attention(q, k, v)
```

## Commit 02: memory bottleneck

Run the naive CPU forward pass at N=128–8192 with fixed B=1, H=1, D=64:

```bash
uv run --locked python -m tileforge.memory_baseline --output /tmp/naive-cpu-baseline.json
```

Use a new output filename for each run. This Linux-only command measures each size in a fresh process, checks its output against SDPA, measures first-call peak process RSS, and records the median of seven forward timings after two warmups. A separate profiler call attributes CPU allocations to operators. Defaults use FP32 and one compute thread; `--lengths`, `--head-dim`, `--threads`, `--warmup`, `--repeats`, and `--seed` are configurable.

The [2026-10-12 observations](observations/2026-10-12/README.md) include the actual results and raw JSON. At N=8192, three full score-sized tensors occupy 768 MiB versus 6 MiB of Q/K/V inputs; measured process peak RSS reaches 1031.79 MiB. These are CPU measurements, not GPU memory or speedup claims.

## Commit 03: online softmax on CPU

[`online_attention`](tileforge/online_softmax.py) handles one query `(D,)` against K/V `(N, D)` in chunks. After processing each chunk, it keeps:

- `m`: the maximum score seen so far.
- `l`: the sum of `exp(score - m)` over all keys seen so far.
- `acc`: the sum of `exp(score - m) * value` over those keys.

When a chunk raises the maximum to `new_m`, multiply both old accumulators by `exp(m - new_m)` before adding the chunk's contributions. This puts old and new contributions on the same scale. The final output is `acc / l`.

The function requires CPU FP32 inputs and finite inputs/scaled scores. It performs forward-only attention with autograd disabled, using O(chunk size + D) scratch space rather than keeping a full sequence of scores. K/V remain resident in memory. Multiple query rows and full CPU tiling are the next milestone.

```bash
uv run --locked python -m tileforge.online_softmax --trace
uv run --locked python -m tileforge.online_softmax --seq-len 17 --chunk-size 4 --scenario rising --trace
uv run --locked python -m tileforge.online_softmax --seq-len 8192 --head-dim 64 --chunk-size 127 --seed 11
```

Each command runs the online implementation and compares it with full softmax attention and PyTorch SDPA, exiting nonzero on a nonfinite output or mismatch. Use `--scenario uniform` for a mean(V) comparison or `--scenario negative` for large negative scores. `--trace` prints the running maximum, rescaling factor, and normalization sum per chunk. See the [actual runs and observations](observations/2026-10-12/online-softmax.md).

## Commit 04: CPU tiled attention

[`tiled_attention`](tileforge/tiled.py) computes every query row from Q/K/V of shape `(N, D)`. It scans query and K/V tiles, maintaining separate running maxima, sums, and numerator vectors per query row. Its CPU FP32 forward pass uses O(Br × Bc + Br × D) scratch space plus resident inputs/output, with autograd disabled.

Partial tiles and non-contiguous inputs are supported. If both requested tiles cover the entire sequence, K/V are split internally to avoid allocating an N × N score tile. N=1 returns V directly. Inputs and scaled scores must be finite. This reference has no batching, masking, or dropout.

```bash
uv run --locked python -m tileforge.tiled
uv run --locked python -m tileforge.tiled --seq-len 65 --head-dim 13 --query-tile 16 --kv-tile 23 --strided --seed 11
uv run --locked python -m tileforge.memory_baseline --implementation tiled --lengths 2048 4096 8192 --skip-profile --output /tmp/tiled-cpu-baseline.json
```

The correctness command compares all output rows with naive attention and SDPA. The memory command isolates its measurements from reference allocations. Choose a new results filename on each run. `--skip-profile` skips only operator profiling, preserving memory, timing, and correctness measurements.

See [commit 04 observations](observations/2026-10-12/tiled-attention.md) for actual commands and raw measurements. At N=8192/D=64, peak process RSS was **265.74 MiB tiled versus 1031.77 MiB naive**; both computed the full output and matched SDPA. These are CPU measurements. All subsequent GPU milestones require real NVIDIA GPU execution.

## Commit 05: Triton GPU softmax

[`triton_softmax`](tileforge/triton_softmax.py) applies row-wise softmax to a contiguous 2D CUDA FP32 tensor. Triton launches one program per row. Each program loads one power-of-two block, masks lanes beyond the real column count, subtracts the row maximum, reduces the exponentials to their sum, and stores only valid outputs. Subtracting the maximum keeps exponentials finite without changing the softmax result.

Run this module in an environment with CUDA-enabled PyTorch, Triton, and a compatible NVIDIA GPU:

```bash
python3 -m tileforge.triton_softmax
python3 -m tileforge.triton_softmax --rows 1 --columns 1
python3 -m tileforge.triton_softmax --rows 128 --columns 4096 --seed 11
```

Each command compares the Triton output with `torch.softmax`, checks finiteness, and exits nonzero on a mismatch. The project lock remains the CPU environment used for commits 00–04; the GPU runs used the RunPod image's CUDA-enabled PyTorch 2.8.0 and Triton 3.4.0 rather than replacing that lock with machine-specific GPU packages.

See [commit 05 observations](observations/2026-10-12/triton-softmax.md) for the exact commands and results from an NVIDIA A100 80 GB. All tested shapes passed, including non-power-of-two widths that exercise masked loads and stores. This is a softmax learning kernel, not attention, and no performance claim is made.

## Commit 06: tiled QKᵀ in Triton

[`triton_qk`](tileforge/triton_qk.py) computes an unscaled score matrix from contiguous 2D CUDA FP32 Q and K tensors. Triton launches a two-dimensional grid of programs. Each program owns one 16×16 output tile, loads Q and K in 16-wide head-dimension chunks, calls `tl.dot`, and masks partial rows, columns, and reduction chunks.

```bash
python3 -m tileforge.triton_qk
python3 -m tileforge.triton_qk --query-rows 33 --key-rows 35 --head-dim 31 --seed 11
python3 -m tileforge.triton_qk --query-rows 128 --key-rows 96 --head-dim 64 --seed 19
```

Each command compares the complete Triton score matrix with `Q @ K.T`, checks finiteness, and exits nonzero on a mismatch. See [commit 06 observations](observations/2026-10-12/triton-qk.md) for the A100 environment, exact commands, and results. All five tested shapes passed; the largest absolute error was `9.53674316e-06`.

This milestone deliberately writes the full score matrix so every local tile can be checked. It does not scale scores, apply softmax, read V, or avoid the N×N HBM allocation. Milestone 07 will fuse those operations and stop writing scores to HBM.

## Commit 07: fused Triton attention

[`triton_attention`](tileforge/triton_attention.py) is the first complete FlashAttention forward engine in this project. One Triton program owns 16 query rows, loads Q once, scans K/V in 16-row tiles, computes scaled scores, updates online-softmax maxima and sums, accumulates weighted V, and writes only the final `(N, D)` output. No full N×N score or weight tensor is allocated in HBM.

```bash
python3 -m tileforge.triton_attention
python3 -m tileforge.triton_attention --seq-len 64 --head-dim 32 --scenario uniform --seed 23
python3 -m tileforge.triton_attention --seq-len 128 --head-dim 64 --scenario rising --seed 29
```

Each command compares the fused output with naive attention and PyTorch SDPA, checks finiteness, and exits nonzero on a mismatch. The uniform scenario must also equal `mean(V)`. The rising scenario makes later K/V tiles raise the row maximum, directly checking that the kernel rescales earlier softmax state correctly.

See [commit 07 observations](observations/2026-10-12/triton-attention.md) for the RTX 2000 Ada environment, exact commands, and results. All six runs passed; the largest absolute error was `4.76837158e-07` against naive attention and `2.02655792e-06` against SDPA.

This first fused kernel is forward-only, non-causal, single-sequence CUDA FP32. N must be a multiple of 16 and D must be 16, 32, or 64. Commit 08 will add tail masks, batches/heads, causal masking, FP16, and BF16.

## Commit 08: Triton correctness features

[`triton_attention`](tileforge/triton_attention.py) now accepts either `(N, D)` or `(B, H, N, D)` Q/K/V tensors, preserves the input shape and dtype, and supports FP32, FP16, and BF16. Masks protect partial query rows, K/V rows, and padded head-dimension lanes. An optional causal mask prevents each query from reading future keys.

```bash
python3 -m tileforge.triton_attention
python3 -m tileforge.triton_attention --batch 2 --heads 3 --seq-len 37 --head-dim 24 --causal --seed 7
python3 -m tileforge.triton_attention --batch 2 --heads 2 --seq-len 67 --head-dim 48 --dtype float16 --seed 11
python3 -m tileforge.triton_attention --batch 1 --heads 3 --seq-len 33 --head-dim 80 --dtype bfloat16 --causal --seed 17
```

The CLI builds an explicit FP32 attention reference, compares PyTorch SDPA using the requested dtype and causal mode, and uses dtype-specific tolerances. See [commit 08 observations](observations/2026-10-12/triton-attention-features.md) for the exact A100 commands and results. All eight checks passed, including 1×1, odd N/D tails, multiple batches/heads, causal masking, uniform weights, rising maxima, and the preserved 2D API.

The supported head dimension is at most 128. Inputs must be contiguous and Q/K/V must share their shape, dtype, and CUDA device. This remains forward-only self-attention without dropout; performance tuning starts in commit 09.

## Milestones

| Commit | Capability | Verification |
| --- | --- | --- |
| 00 | Repository, Python environment, hardware diagnostic | CPU PyTorch smoke check |
| 01 | Naive PyTorch attention, starting at B=1, H=1, N=4, D=4 | Matches SDPA |
| 02 | N × N score matrix and runtime memory measurements | Baseline measurements |
| 03 | Online softmax on CPU for one query | Matches full softmax attention |
| 04 | CPU tiled attention | Correct without a full score matrix |
| 05 | Triton GPU softmax | Correct loads, stores, masks, reductions |
| 06 | Tiled QKᵀ in Triton | Correct tile products |
| 07 | Fused Triton attention | First FlashAttention forward kernel |
| 08 | Causal masks, tail tiles, batches/heads, FP16/BF16 | Correct on supported shapes |
| 09 | Triton tile, warp, and pipeline tuning | Recorded performance changes |
| 10 | PyTorch CUDA C++ extension | Callable CUDA kernel |
| 11 | Basic CUDA attention | Correct numerical baseline |
| 12 | Fused CUDA attention | Correct shared-memory tiled kernel |
| 13 | Benchmarks and documentation | Reproducible comparisons and plots |

Final benchmarks compare naive PyTorch, automatic SDPA, explicitly selected FlashAttention SDPA where supported, Triton, and CUDA. Use identical batch size, heads, head dimension, dtype, and causal mode at N = 128, 256, 512, 1024, 2048, 4096. Report error against an FP32 reference, median warmed-up CUDA-event latency, peak allocated GPU memory, and speedup defined as SDPA latency / custom latency.

## References

- [Triton fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Triton matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [NVIDIA CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
