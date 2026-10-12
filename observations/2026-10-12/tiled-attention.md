# 2026-10-12 — Commit 04: CPU tiled attention

Implemented `tiled_attention(q, k, v, query_tile_size, kv_tile_size)` for full `(N, D)` CPU FP32 inputs. It scans Q tiles and K/V tiles, keeping a separate maximum, normalization sum, and weighted numerator for every query row. It writes each completed output tile directly into the `(N, D)` output.

Scratch space is O(Br × Bc + Br × D), excluding resident Q/K/V and output. Autograd is disabled. The implementation does not allocate a full N × N score matrix: when both requested tile dimensions cover the entire sequence, it splits K/V; N=1 returns V directly. No batching, causal mask, reduced precision, or GPU kernels are included in this milestone.

## Manual correctness runs

[tiled-attention-runs.txt](tiled-attention-runs.txt) preserves nine commands, outputs, timestamps, and exit codes. Each command runs `python -m tileforge.tiled` and compares its full output with naive attention and SDPA. All exited 0 at `rtol=1e-5, atol=1e-6`.

| N | D | Requested Q/KV tile sizes | Scenario | Max error vs naive | Max error vs SDPA |
| ---: | ---: | --- | --- | ---: | ---: |
| 17 | 8 | 4 / 8 | Both axes have incomplete final tiles | 3.57627869e-7 | 3.57627869e-7 |
| 4 | 4 | 1 / 1 | Single-row and single-key tiles | 2.38418579e-7 | 4.09781933e-8 |
| 9 | 7 | 32 / 32 | Oversized tiles; K/V split internally | 2.38418579e-7 | 1.19209290e-7 |
| 1 | 1 | 4 / 8 | Single value, returned directly | 0 | 0 |
| 33 | 1 | 7 / 11 | One-dimensional heads | 5.96046448e-8 | 4.47034836e-8 |
| 17 | 4 | 4 / 8 | Positive/negative extreme scores and a zero-score row | 3.57627869e-7 | 1.19209290e-7 |
| 17 | 8 | 4 / 8 | Uniform scores; mean(V) also verified | 5.96046448e-8 | 5.96046448e-8 |
| 65 | 13 | 16 / 23 | Non-contiguous inputs, partial tiles | 2.98023224e-7 | 2.98023224e-7 |
| 1024 | 64 | 64 / 128 | Larger full-sequence output | 2.68220901e-7 | 1.56462193e-7 |

The extreme scenario uses positive, negative, and zero query coefficients within the same Q tile. This exercises per-row maxima and broadcasting rather than sharing one normalization statistic across rows.

## Full-sequence memory and timing comparison

Fresh runs started at 11:13 Asia/Kolkata on 2026-10-12. Both implementations compute **all N query rows**, with identical random inputs, seed 0, B=1, H=1, D=64, FP32, no autograd, no dropout, and no causal mask. Tiled uses Br=64 and Bc=128. Python 3.12.14, PyTorch 2.10.0+cpu, Intel Xeon Platinum 8573C; one compute thread and one inter-op thread, within a four-CPU/32 GiB container.

```bash
UV_CACHE_DIR=/workspace/.cache/uv uv run --locked python -m tileforge.memory_baseline \
  --implementation naive --lengths 2048 4096 8192 --skip-profile \
  --output observations/2026-10-12/tiled-comparison-naive.json

UV_CACHE_DIR=/workspace/.cache/uv uv run --locked python -m tileforge.memory_baseline \
  --implementation tiled --lengths 2048 4096 8192 --query-tile 64 --kv-tile 128 \
  --skip-profile --output observations/2026-10-12/tiled-comparison-tiled.json
```

Both commands exited 0; all six full outputs matched SDPA. Use new filenames to repeat the measurements. Raw files retain metadata and all timing samples: [naive](tiled-comparison-naive.json), [tiled](tiled-comparison-tiled.json).

| N | Naive peak RSS MiB | Tiled peak RSS MiB | Naive first-call HWM increase MiB | Tiled first-call HWM increase MiB | Naive median ms | Tiled median ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 304.66 | 259.22 | 61.18 | 15.74 | 35.688 | 34.967 |
| 4096 | 450.25 | 260.94 | 206.18 | 16.30 | 170.991 | 142.015 |
| 8192 | 1031.77 | 265.74 | 784.12 | 17.36 | 606.434 | 542.815 |

At N=8192, naive attention's three full score-sized intermediates total 768 MiB. Each tiled score buffer contains at most 64 × 128 FP32 elements, **32 KiB**; this is one buffer's payload, not the entire working set. The accumulator holds 64 × 64 FP32 elements, **16 KiB**, plus row statistics, other tile temporaries, and library overhead. Q/K/V (6 MiB combined) and the final output (2 MiB) remain resident.

The measured first-call high-water increase falls from 784.12 to 17.36 MiB. Peak RSS is about 766 MiB lower. The memory advantage comes from limiting live intermediates; the arithmetic still scales quadratically in N.

Timing modestly favored tiling in this run, but this is not an optimized CPU kernel or a GPU speedup claim. At N=8192, naive samples ranged from 572.811 to 798.015 ms; tiled samples ranged from 526.309 to 559.610 ms. Allocation, cache effects, Python loops, matrix-operation dispatch, and shared-machine noise all affect these measurements. No hardware bandwidth counters were collected.

## Measurement boundaries and allocation profiling

Each implementation/length runs in a fresh process. Memory is captured immediately around the first forward call, after input allocation and before correctness references or warmups. Linux `ru_maxrss` records process lifetime high-water RSS, not exact live tensor allocations; subtraction includes first-use runtime overhead and can hide allocations below an earlier high-water mark. Timings use two warmups and the median of seven synchronous CPU forward calls, excluding SDPA, input creation, printing, and profiler overhead.

`--skip-profile` avoids retaining thousands of profiler events during the large experiments. It does not skip RSS measurements, timings, or SDPA comparisons. In the raw JSON, `one_score_matrix_bytes` is the hypothetical dense payload for either implementation; it does not imply tiled attention allocated it. `three_live_score_matrices_bytes` and profiling results are null where not applicable or not collected.

A separate allocation run executed:

```bash
UV_CACHE_DIR=/workspace/.cache/uv uv run --locked python -m tileforge.memory_baseline \
  --implementation tiled --lengths 257 --query-tile 16 --kv-tile 32 \
  --output observations/2026-10-12/tiled-allocation-profile.json
```

It exited 0, matched SDPA with maximum error `3.87430191e-7`, and recorded **306 matrix multiplications**: 17 query tiles × 9 K/V tiles × 2 matrix products. The [allocation records](tiled-allocation-profile.json) report cumulative operator-attributed bytes, not peak live memory. The profiler issued the same single-cycle event-clearing warning noted in commit 02; the expected events were present. The benchmark's default naive profiling path was also run at N=128 and passed.

The correctness CLI intentionally allocates full naive references outside `tiled_attention`; these are absent from the measured first-call memory interval. No unit tests were created, modified, or run.

## Next: real GPU access

**05 — Triton GPU softmax** requires an accessible NVIDIA GPU and CUDA-enabled PyTorch/Triton. It remains unchecked. The current CPU-only environment cannot validate it; do not substitute CPU execution for GPU results.
