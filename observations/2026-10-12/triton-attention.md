# Commit 07: fused Triton attention

Milestone 07 produced the project's first real FlashAttention forward engine: it computes scaled QKᵀ, online softmax, and weighted V in one Triton kernel without writing a full score or weight matrix to GPU HBM.

## Normal flow

1. Python asks PyTorch to allocate contiguous CUDA FP32 Q, K, and V matrices with shape `(N, D)`.
2. `triton_attention` asks Triton to launch one program for every 16 query rows.
3. Each program asks `tl.load` for its Q tile once and creates per-row running maxima, normalization sums, and a 16×D weighted-V accumulator.
4. The program asks `tl.load` for the next 16 K rows and 16 V rows, then asks `tl.dot` to calculate one local QKᵀ score tile.
5. The program scales the scores by `1/√D`, compares the tile maximum with the running maximum, and rescales earlier sums and weighted-V values when the maximum increases.
6. The program adds the current tile's exponentials to the normalization sum and asks `tl.dot` to add the current probability-times-V contribution to the accumulator.
7. The program repeats the K/V request until it has scanned the sequence, divides the accumulator by the normalization sum, and writes only the final output tile.
8. Python asks naive attention and PyTorch SDPA for reference outputs, synchronizes the GPU, and returns success only when the fused result is finite and close to both.

## Environment and commands

Runs occurred at 2026-10-12 21:50 Asia/Kolkata from `/workspace/tileforge` on RunPod:

- NVIDIA RTX 2000 Ada Generation, 16 GB, compute capability 8.9
- NVIDIA driver 580.159.04
- Python 3.12.3
- PyTorch 2.8.0+cu128
- Triton 3.4.0
- `nvcc` unavailable; it is not required until milestone 10

The environment diagnostic and six fused-kernel configurations all exited 0:

```bash
python3 scripts/check_environment.py --require-cuda
python3 -m tileforge.triton_attention
python3 -m tileforge.triton_attention --seq-len 16 --head-dim 16 --seed 7
python3 -m tileforge.triton_attention --seq-len 32 --head-dim 64 --seed 11
python3 -m tileforge.triton_attention --seq-len 128 --head-dim 64 --seed 19
python3 -m tileforge.triton_attention --seq-len 64 --head-dim 32 --scenario uniform --seed 23
python3 -m tileforge.triton_attention --seq-len 128 --head-dim 64 --scenario rising --seed 29
```

## Results

| N | D | Scenario | K/V tiles scanned | Max error vs naive | Max error vs SDPA |
| ---: | ---: | --- | ---: | ---: | ---: |
| 16 | 16 | random | 1 | 1.78813934e-7 | 3.57627869e-7 |
| 32 | 64 | random | 2 | 4.76837158e-7 | 2.02655792e-6 |
| 64 | 32 | random | 4 | 2.68220901e-7 | 1.25169754e-6 |
| 128 | 64 | random | 8 | 4.17232513e-7 | 7.74860382e-7 |
| 64 | 32 | uniform | 4 | 2.98023224e-8 | 5.96046448e-8 |
| 128 | 64 | rising | 8 | 1.19209290e-7 | 1.37090683e-6 |

Every output was finite, had shape `(N, D)`, and matched naive attention and SDPA at `rtol=1e-4, atol=1e-5`. Uniform scores also matched `mean(V)` with maximum error `4.09781933e-8`.

The rising case made scores increase from 0 to 20 as the kernel moved through eight K/V tiles. Later tiles repeatedly replaced the running maximum, so passing this case directly checks the `alpha = exp(old_max - new_max)` rescaling of both earlier normalization sums and weighted-V accumulators.

The implementation structurally avoids full N×N HBM intermediates: Python allocates only the `(N, D)` output, and the kernel never stores its 16×16 score tile through an output pointer. Triton manages the tile's local storage. No GPU allocation profiling or compiler-spill inspection was performed, so this is not a measured-memory result.

## Limits and next step

This forward-only kernel supports one contiguous, non-causal CUDA FP32 sequence. N must be divisible by 16, and D must be 16, 32, or 64. It does not support tail tiles, batches, heads, causal masking, FP16, BF16, gradients, dropout, or performance claims.

Milestone 08 will add causal masking, non-multiple tile sizes, multiple batches/heads, FP16, and BF16 before performance tuning begins.
