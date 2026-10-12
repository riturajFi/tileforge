# Triton attention tuning observations

## Core result

On the RunPod A100 workload measured here, 64×64 query/key tiles with four warps and two pipeline stages were the fastest tested configuration. This tuple is now the default used by `triton_attention`.

## How the tuning ran

`tileforge.triton_tuning` asks the existing fused kernel to run with one candidate launch configuration. Before timing it, the command asks PyTorch SDPA for the same output and rejects the candidate if the result is non-finite or outside FP16 tolerances. It then performs 10 warmups and records 30 CUDA-event samples, reporting their median.

The fixed workload was non-causal FP16 with B=1, H=8, and D=64. The command measured N=128, 256, 512, and 1024 and ranked each configuration by the geometric mean of its four median latencies. It first changed tile dimensions, kept the winning tiles while changing warp count, then kept that winner while changing pipeline stages.

## What the numbers mean

| Choice | Geometric-mean latency | Meaning |
| --- | ---: | --- |
| Old 16×16 tiles, 4 warps, 2 stages | 0.126402 ms | Starting configuration |
| 64×64 tiles, 4 warps, 2 stages | 0.113152 ms | Tile winner; 10.5% lower than the starting configuration |
| 64×64 tiles, 2 warps, 2 stages | 0.126502 ms | Too few warps for this workload |
| 64×64 tiles, 4 warps, 2 stages | 0.111288 ms | Warp-count winner in its sweep |
| 64×64 tiles, 8 warps, 2 stages | 0.122177 ms | Extra warps did not recover their overhead |
| 64×64 tiles, 4 warps, 2 stages | 0.112992 ms | Pipeline-stage winner in its sweep |
| 64×64 tiles, 4 warps, 3 stages | 0.114883 ms | Slightly slower |
| 64×64 tiles, 4 warps, 4 stages | 0.113173 ms | Slightly slower |

Larger tiles helped most at N=1024: the tile sweep measured `0.172576 ms` for 16×16 and `0.116240 ms` for 64×64. The small difference between two and four pipeline stages may be timing noise, so the conclusion is limited to this exact sweep rather than a universal GPU rule.

Every candidate matched SDPA before timing. Across the sweep, the largest absolute error was `4.88281250e-4`. After installing the winner as the default, separate FP32, FP16, and BF16 checks also passed for causal masking, odd sequence/head dimensions, multiple batches/heads, rising softmax maxima, and N=D=1.

## Limits

These are single-machine kernel measurements from an NVIDIA A100 80 GB PCIe. They do not prove that this tuple is fastest on another GPU, dtype, head dimension, causal mode, or batch/head count. They also do not compare Triton latency with SDPA, measure GPU memory, or claim an application-level speedup; milestone 13 owns those controlled comparisons. No unit tests were run, written, or modified.

The exact command, raw measurements, environment check, and post-tuning correctness output are in `triton-attention-tuning-runs.txt`.
