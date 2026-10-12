# Commit 08: Triton attention correctness features

Milestone 08 makes the fused Triton kernel usable across the required attention shapes and dtypes: causal and non-causal execution, tail tiles, batches, heads, FP32, FP16, and BF16 all passed on a real A100.

## Normal flow

1. Python asks PyTorch to allocate Q, K, and V as either `(N, D)` or `(B, H, N, D)` on one CUDA device.
2. `triton_attention` validates the shared shape, dtype, device, contiguity, causal flag, and maximum supported head dimension, then flattens batch and head into one launch dimension.
3. Python asks Triton to launch programs across every batch/head pair and every 16-row query tile.
4. Each program asks `tl.load` for Q/K/V tiles. Masks replace missing tail rows and padded head-dimension lanes with zero.
5. The program computes local scores. A key mask replaces positions beyond N with negative infinity; causal mode also replaces every future-key position with negative infinity.
6. The program updates FP32 online-softmax state and weighted-V accumulators. For FP16 or BF16 V, it converts the local probabilities to the input dtype for `tl.dot`, while keeping accumulated results in FP32.
7. The program scans every K/V tile, normalizes the accumulated output, and asks `tl.store` to write only valid query rows and head-dimension lanes in the original dtype.
8. Python restores the original 2D or 4D shape, builds an explicit FP32 reference and a matching SDPA result, and returns success only when the output is finite and within the dtype-specific tolerance.

## Environment and commands

Runs occurred at 2026-10-12 22:01 Asia/Kolkata from `/workspace/tileforge` on RunPod:

- NVIDIA A100 80GB PCIe, compute capability 8.0, BF16 supported
- NVIDIA driver 580.159.04
- Python 3.12.3
- PyTorch 2.8.0+cu128
- Triton 3.4.0
- `nvcc` unavailable; it is not required until milestone 10

The environment diagnostic and eight direct kernel checks all exited 0. Exact console output is in [triton-attention-features-runs.txt](triton-attention-features-runs.txt).

## Results

| Shape | Dtype | Causal | Extra check | Max error vs FP32 reference | Max error vs SDPA |
| --- | --- | --- | --- | ---: | ---: |
| `(1,1,65,40)` | FP32 | No | Q/K/V and D tails | 5.96046448e-7 | 8.04662704e-7 |
| `(2,3,37,24)` | FP32 | Yes | Multiple batches/heads | 5.36441803e-7 | 1.25169754e-6 |
| `(2,2,67,48)` | FP16 | No | Multiple batches/heads | 4.01735306e-4 | 4.88281250e-4 |
| `(1,3,33,80)` | BF16 | Yes | D spans padded 128 lanes | 5.88369370e-3 | 3.90625000e-3 |
| `(2,2,17,23)` | FP16 | Yes | Rising maxima | 9.54389572e-4 | 4.88281250e-4 |
| `(2,3,19,7)` | BF16 | No | Uniform equals mean(V) | 8.99493694e-4 | 0 |
| `(1,1,1,1)` | FP32 | Yes | Smallest boundary | 0 | 0 |
| `(17,23)` | FP32 | Yes | Preserved 2D API | 1.78813934e-7 | not separately recorded |

All FP32 runs passed at `rtol=1e-4, atol=1e-5`; FP16 at `rtol=2e-2, atol=2e-3`; BF16 at `rtol=3e-2, atol=1e-2`. Lower-precision errors are larger because inputs, local probability-times-V products, and final outputs use fewer representable values, while online-softmax state and accumulation remain FP32.

## Limits and next step

The kernel remains forward-only self-attention with identical Q/K/V shapes, no dropout, contiguous inputs, and D ≤ 128. It does not support gradients, non-contiguous layouts, different query/key sequence lengths, arbitrary attention masks, or performance claims.

Milestone 09 will measure the correct kernel, then tune tile sizes, warp counts, and pipeline stages without broadening correctness scope.
