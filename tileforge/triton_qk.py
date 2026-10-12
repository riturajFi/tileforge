"""Tiled QK-transpose products using Triton on an NVIDIA GPU."""

import argparse

import torch
import triton
import triton.language as tl

from tileforge.naive import positive_int


@triton.jit
def _qk_kernel(q_ptr, k_ptr, output_ptr,
               q_rows, k_rows,
               q_row_stride, k_row_stride, output_row_stride,
               HEAD_DIM: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_K: tl.constexpr):
    q_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction_offsets = tl.arange(0, BLOCK_K)
    scores = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for start in range(0, HEAD_DIM, BLOCK_K):
        dimensions = start + reduction_offsets
        q = tl.load(
            q_ptr + q_offsets[:, None] * q_row_stride + dimensions[None, :],
            mask=(q_offsets[:, None] < q_rows) & (dimensions[None, :] < HEAD_DIM),
            other=0.0,
        )
        k = tl.load(
            k_ptr + k_offsets[:, None] * k_row_stride + dimensions[None, :],
            mask=(k_offsets[:, None] < k_rows) & (dimensions[None, :] < HEAD_DIM),
            other=0.0,
        )
        scores += tl.dot(q, tl.trans(k), input_precision="ieee")

    output_offsets = q_offsets[:, None] * output_row_stride + k_offsets[None, :]
    output_mask = (q_offsets[:, None] < q_rows) & (k_offsets[None, :] < k_rows)
    tl.store(output_ptr + output_offsets, scores, mask=output_mask)


@torch.no_grad()
def triton_qk(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Return QKᵀ for contiguous 2D CUDA FP32 tensors without scaling."""
    if q.ndim != 2 or k.ndim != 2 or q.shape[1:] != k.shape[1:] or 0 in q.shape or 0 in k.shape:
        raise ValueError("Expected nonempty Q and K with shapes (M, D) and (N, D).")
    if q.device != k.device or q.device.type != "cuda" or q.dtype != torch.float32 or k.dtype != q.dtype:
        raise ValueError("This milestone requires CUDA FP32 Q and K on the same device.")
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("Q and K must be contiguous.")

    q_rows, head_dim = q.shape
    k_rows = k.shape[0]
    output = q.new_empty((q_rows, k_rows))
    block_m = block_n = block_k = 16
    grid = (triton.cdiv(q_rows, block_m), triton.cdiv(k_rows, block_n))
    _qk_kernel[grid](
        q, k, output,
        q_rows, k_rows,
        q.stride(0), k.stride(0), output.stride(0),
        HEAD_DIM=head_dim,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-rows", type=positive_int, default=17)
    parser.add_argument("--key-rows", type=positive_int, default=23)
    parser.add_argument("--head-dim", type=positive_int, default=13)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA is unavailable.")
        return 1

    torch.manual_seed(args.seed)
    q = torch.randn(args.query_rows, args.head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(args.key_rows, args.head_dim, device="cuda", dtype=torch.float32)
    output = triton_qk(q, k)
    reference = q @ k.T
    torch.cuda.synchronize()

    max_error = (output - reference).abs().max().item()
    close = torch.allclose(output, reference, rtol=1e-5, atol=1e-5)
    finite = bool(torch.isfinite(output).all())
    passed = close and finite
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch {torch.__version__}; Triton {triton.__version__}; "
          f"Q={tuple(q.shape)}; K={tuple(k.shape)}; CUDA FP32; seed={args.seed}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Maximum absolute error vs Q @ K.T: {max_error:.8e}")
    print(f"{'PASS' if passed else 'FAIL'}: finite output and reference agreement "
          "(rtol=1e-5, atol=1e-5)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
