# Tileforge

Build FlashAttention forward-pass kernels from scratch in **Triton** and **CUDA C++**, then compare correctness, latency, and GPU memory with PyTorch scaled dot-product attention (SDPA).

**Small theory → small commit → verify → next commit.**

## Commit 00: reproducible CPU environment

Naive attention materializes an N × N score matrix. FlashAttention instead scans Q/K/V tiles, keeps running softmax statistics and output accumulators, and avoids writing the full score matrix to GPU memory. We will build and verify those pieces incrementally. No custom attention implementation is included yet.

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
