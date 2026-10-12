# Commit 05: Triton GPU softmax

Milestone 05 ran on a real NVIDIA GPU and passed: one Triton program computed each softmax row, including masked tail lanes for widths that are not powers of two.

## Normal flow

1. Python asks PyTorch to allocate a contiguous FP32 matrix on the CUDA device.
2. `triton_softmax` asks Triton to launch one program for each matrix row.
3. Each Triton program asks `tl.program_id(0)` which row it owns.
4. The program asks `tl.load` for a power-of-two block from that row. A mask excludes offsets beyond the real column count, and excluded lanes receive negative infinity.
5. Triton reduces the block to its maximum. The program subtracts that maximum from every loaded value so large positive or negative inputs remain numerically stable.
6. Triton exponentiates the shifted values and reduces them to one row sum.
7. The program divides each exponential by that sum and asks `tl.store` to write only the valid columns.
8. Python asks PyTorch to compute the same rows with `torch.softmax`, synchronizes the GPU, and returns success only if the Triton result is finite and close to the reference.

## Environment and commands

Runs occurred at 2026-10-12 20:45 Asia/Kolkata from `/workspace/tileforge` on RunPod:

- NVIDIA A100 80GB PCIe, compute capability 8.0
- NVIDIA driver 580.159.04
- Python 3.12.3
- PyTorch 2.8.0+cu128
- Triton 3.4.0
- `nvcc` unavailable; it is not required until milestone 10

The environment diagnostic and four module configurations all exited 0:

```bash
python3 scripts/check_environment.py --require-cuda
python3 -m tileforge.triton_softmax
python3 -m tileforge.triton_softmax --rows 1 --columns 1
python3 -m tileforge.triton_softmax --rows 3 --columns 17 --seed 7
python3 -m tileforge.triton_softmax --rows 128 --columns 4096 --seed 11
```

A direct Python command also used input rows near +10000 and -10000 to exercise max-subtracted exponentials. It exited 0 and produced finite output.

## Results

| Rows × columns | Why it ran | Maximum error vs PyTorch | Maximum row-sum error |
| ---: | --- | ---: | ---: |
| 1 × 1 | Smallest valid matrix | 0 | 0 |
| 3 × 17 | Small non-power-of-two mask | 2.98023224e-8 | 0 |
| 7 × 513 | Default non-power-of-two mask | 1.86264515e-9 | 1.19209290e-7 |
| 128 × 4096 | Larger reduction | 1.86264515e-9 | 2.38418579e-7 |
| 2 × 3 extreme | Numerical stability near ±10000 | 5.96046448e-8 | not separately recorded |

Every output was finite and matched `torch.softmax` at `rtol=1e-5, atol=1e-6`. Exact console output and exit codes are in [triton-softmax-runs.txt](triton-softmax-runs.txt).

## Limits and next step

This forward-only learning kernel accepts nonempty, contiguous, 2D CUDA FP32 tensors. It does not support gradients, FP16, BF16, non-contiguous layouts, attention, or performance claims. Those features are outside milestone 05.

Milestone 06 will load Q and K tiles in Triton, compute local QKᵀ scores with `tl.dot`, and compare them with PyTorch before adding online softmax or V accumulation.
