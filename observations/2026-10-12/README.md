# 2026-10-12 — Commit 02: naive attention memory bottleneck

Actual run started at **10:37:46 Asia/Kolkata**. Raw measurements, all timing samples, hardware metadata, and profiler allocation records are in [naive-cpu-baseline.json](naive-cpu-baseline.json).

## What ran

```bash
UV_CACHE_DIR=/workspace/.cache/uv uv run --locked python -m tileforge.memory_baseline \
  --output observations/2026-10-12/naive-cpu-baseline.json
```

The command finished with exit 0. It executed the existing `naive_attention` from commit `435cddf`, without modifying that implementation. Every sequence length also passed comparison with PyTorch SDPA (`rtol=1e-5`, `atol=1e-6`). Maximum absolute error across the seven shapes was `1.1920928955078125e-6`; the relative tolerance accounts for that value exceeding the absolute tolerance alone.

Configuration: CPU FP32, B=1, H=1, D=64, seed 0, no causal mask, no dropout, no autograd. Python 3.12.14, PyTorch 2.10.0+cpu, Intel Xeon Platinum 8573C. The container has a four-CPU quota and 32 GiB memory limit; this experiment deliberately uses **one PyTorch compute thread** and one inter-op thread. No GPU measurements were made.

## Actual results

MiB means 1,048,576 bytes. Matrix sizes are exact tensor payload calculations for the executed shapes, corroborated by operator allocation records. RSS and timing columns are runtime measurements.

| N | Q/K/V total MiB | One N × N matrix MiB | Three live matrices MiB | Process peak RSS MiB | First-call HWM increase MiB | Median latency ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.09375 | 0.0625 | 0.1875 | 253.21 | 10.75 | 0.080 |
| 256 | 0.1875 | 0.25 | 0.75 | 254.54 | 12.68 | 0.247 |
| 512 | 0.375 | 1 | 3 | 257.26 | 15.37 | 1.528 |
| 1024 | 0.75 | 4 | 12 | 267.01 | 24.62 | 8.106 |
| 2048 | 1.5 | 16 | 48 | 304.12 | 61.18 | 32.996 |
| 4096 | 3 | 64 | 192 | 450.91 | 206.18 | 151.153 |
| 8192 | 6 | 256 | 768 | 1031.79 | 784.25 | 615.508 |

## Observed bottlenecks

1. **Attention intermediates grow quadratically while inputs grow linearly.** From N=4096 to 8192, Q/K/V grow from 3 to 6 MiB, but one score matrix grows from 64 to 256 MiB. At N=8192, the three score-sized intermediates consume **768 MiB**, or **128 times the combined input payload**.
2. **This implementation keeps three full matrices alive.** `scores = Q @ Kᵀ`, the out-of-place scaling result `scaled_scores`, and softmax `weights` remain referenced until the function returns. At N=8192, the profiler attributes 258 MiB to the two `aten::bmm` calls combined (256 MiB scores + 2 MiB output), 256 MiB minus 4 bytes to `aten::div`, 4 bytes to its scalar allocation, and 256 MiB to `aten::_softmax`. Their positive self-memory values sum to **770 MiB**. These are attributed allocation amounts, not a separately measured peak-live-tensor counter.
3. **The process pays the allocation cost during actual execution.** At N=8192, the first forward call raises the process memory high-water mark by **784.25 MiB**, reaching **1031.79 MiB** total peak RSS. This is consistent with the large temporary tensors plus output and runtime overhead. Merely reporting the 6 MiB input size would hide the main memory cost.
4. **Latency also rises sharply.** Doubling N from 4096 to 8192 increases median forward latency from 151.153 ms to 615.508 ms, approximately **4.07×**. This is consistent with quadratic attention work and larger intermediates. It does **not** establish whether arithmetic, allocation, cache behavior, or memory bandwidth dominates execution; no hardware bandwidth counters were measured.

## Measurement method and limits

- Each N runs in a fresh Python subprocess. Inputs are allocated first; current RSS and the process high-water mark are recorded; one naive forward call runs; RSS and high-water mark are read immediately afterward. SDPA, warmups, timings, and profiling happen only after that memory measurement.
- Peak RSS uses Linux `resource.getrusage(...).ru_maxrss`. HWM increase is the after-minus-before difference between lifetime high-water marks, **not** an exact tensor allocation counter. Python imports, allocator behavior, first-use library work, and Q/K/V contribute to absolute RSS. Prior high-water marks can also hide small temporary increases. Current RSS before and after is preserved in the JSON for context.
- Timing uses `perf_counter_ns` around synchronous CPU forward calls after two warmups, with the median of seven samples. Input generation, SDPA validation, result printing, and profiler overhead are excluded. Output allocation is included; output deletion occurs after each timed region. Memory uses the first call while timing measures warmed-up calls.
- This is one run per shape on a shared cloud CPU. All seven timing samples per shape are retained. At N=8192 they ranged from 597.813 to 750.061 ms, so medians are more representative than a single sample.
- A separate CPU profiler call records operator allocations after timing. PyTorch emitted a warning about clearing events between profiling cycles; this experiment uses a single cycle per process, and its expected matrix/output allocations were present.
- These results demonstrate CPU allocation growth, not GPU HBM usage or GPU FlashAttention speedup. No unit tests were created, changed, or run.

## Next step

**03 — Online softmax on CPU:** process one query against K in chunks while maintaining a running maximum, normalization sum, and weighted numerator. Verify the output against full softmax attention. The later tiled implementation will use that recurrence to avoid full N × N intermediates.

To rerun, choose a new output filename: the command deliberately refuses to overwrite existing observations.
