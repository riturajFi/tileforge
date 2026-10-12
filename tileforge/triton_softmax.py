"""Row-wise softmax using one Triton GPU program per input row."""

import argparse

import torch
import triton
import triton.language as tl

from tileforge.naive import positive_int


@triton.jit
def _softmax_kernel(input_ptr, output_ptr, input_row_stride, output_row_stride,
                    columns: tl.constexpr, block_size: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < columns
    values = tl.load(input_ptr + row * input_row_stride + offsets,
                     mask=mask, other=-float("inf"))
    values -= tl.max(values, axis=0)
    numerators = tl.exp(values)
    result = numerators / tl.sum(numerators, axis=0)
    tl.store(output_ptr + row * output_row_stride + offsets, result, mask=mask)


@torch.no_grad()
def triton_softmax(inputs: torch.Tensor) -> torch.Tensor:
    """Apply forward-only softmax to contiguous CUDA FP32 rows."""
    if inputs.ndim != 2 or 0 in inputs.shape:
        raise ValueError("Expected a nonempty 2D tensor.")
    if inputs.device.type != "cuda" or inputs.dtype != torch.float32:
        raise ValueError("This milestone requires CUDA float32 inputs.")
    if not inputs.is_contiguous():
        raise ValueError("Input must be contiguous.")

    rows, columns = inputs.shape
    output = torch.empty_like(inputs)
    block_size = triton.next_power_of_2(columns)
    _softmax_kernel[(rows,)](
        inputs, output, inputs.stride(0), output.stride(0), columns, block_size
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=positive_int, default=7)
    parser.add_argument("--columns", type=positive_int, default=513)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA is unavailable.")
        return 1

    torch.manual_seed(args.seed)
    inputs = torch.randn(args.rows, args.columns, device="cuda", dtype=torch.float32)
    output = triton_softmax(inputs)
    reference = torch.softmax(inputs, dim=1)
    torch.cuda.synchronize()

    max_error = (output - reference).abs().max().item()
    max_row_sum_error = (output.sum(dim=1) - 1).abs().max().item()
    close = torch.allclose(output, reference, rtol=1e-5, atol=1e-6)
    finite = bool(torch.isfinite(output).all())
    passed = close and finite
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch {torch.__version__}; Triton {triton.__version__}; "
          f"shape=({args.rows}, {args.columns}); CUDA FP32; seed={args.seed}")
    print(f"Maximum absolute error vs torch.softmax: {max_error:.8e}")
    print(f"Maximum row-sum error: {max_row_sum_error:.8e}")
    print(f"{'PASS' if passed else 'FAIL'}: finite output and reference agreement "
          "(rtol=1e-5, atol=1e-6)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
