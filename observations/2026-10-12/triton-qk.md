# Commit 06: tiled QKᵀ in Triton

Milestone 06 ran on a real NVIDIA GPU and passed: a two-dimensional Triton program grid computed the complete QKᵀ score matrix from independent 16×16 tiles.

## Normal flow

1. Python asks PyTorch to allocate contiguous CUDA FP32 Q and K matrices with the same head dimension.
2. `triton_qk` asks Triton to launch one program for every 16-row Q tile and 16-row K tile pair.
3. Each program asks its two program IDs which Q tile and K tile it owns, then creates a 16×16 FP32 accumulator.
4. The program asks `tl.load` for 16 head-dimension values from each tile. Masks replace missing rows and head-dimension tail values with zero.
5. The program asks `tl.dot` to multiply the Q chunk by the transposed K chunk and add the local products to its accumulator. It repeats until the full head dimension has been scanned.
6. The program asks `tl.store` to write its local scores into the correct part of the output matrix. A mask prevents partial edge tiles from writing outside the matrix.
7. Python asks PyTorch to compute `Q @ K.T`, synchronizes the GPU, and returns success only when the complete Triton matrix is finite, has the expected shape, and matches the reference.

## Environment and commands

Runs occurred at 2026-10-12 21:20 Asia/Kolkata from `/workspace/tileforge` on RunPod:

- NVIDIA A100 80GB PCIe, compute capability 8.0
- NVIDIA driver 595.91.07
- Python 3.12.3
- PyTorch 2.8.0+cu128
- Triton 3.4.0
- `nvcc` unavailable; it is not required until milestone 10

The environment diagnostic and five kernel configurations all exited 0:

```bash
python3 scripts/check_environment.py --require-cuda
python3 -m tileforge.triton_qk
python3 -m tileforge.triton_qk --query-rows 1 --key-rows 1 --head-dim 1
python3 -m tileforge.triton_qk --query-rows 32 --key-rows 48 --head-dim 16 --seed 7
python3 -m tileforge.triton_qk --query-rows 33 --key-rows 35 --head-dim 31 --seed 11
python3 -m tileforge.triton_qk --query-rows 128 --key-rows 96 --head-dim 64 --seed 19
```

## Results

| Q shape | K shape | What it checks | Maximum error |
| ---: | ---: | --- | ---: |
| 1×1 | 1×1 | Smallest input; all three tail masks exercised | 0 |
| 17×13 | 23×13 | Partial Q, K, and head-dimension tiles | 0 |
| 32×16 | 48×16 | Aligned tiles | 0 |
| 33×31 | 35×31 | Partial tiles and two reduction chunks | 3.81469727e-6 |
| 128×64 | 96×64 | Multiple tiles and four reduction chunks | 9.53674316e-6 |

Every output was finite, had shape `(Q rows, K rows)`, and matched `Q @ K.T` at `rtol=1e-5, atol=1e-5`. The small nonzero errors on multi-chunk reductions come from adding FP32 products in a different order from PyTorch. Exact console output and exit codes are in [triton-qk-runs.txt](triton-qk-runs.txt).

## Limits and next step

This forward-only learning kernel accepts nonempty, contiguous, 2D CUDA FP32 tensors. It materializes the complete score matrix in GPU HBM and does not apply attention scaling, softmax, or V. It makes no latency, memory-saving, or speedup claim.

Milestone 07 will combine tiled QKᵀ, scaling, online softmax, and weighted V accumulation in one kernel so score tiles remain on-chip instead of being written to HBM.
