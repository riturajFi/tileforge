"""First fused Triton attention forward kernel with online softmax."""

import argparse
import math

import torch
import torch.nn.functional as functional
import triton
import triton.language as tl

from tileforge.naive import naive_attention, positive_int


@triton.jit
def _attention_kernel(q_ptr, k_ptr, v_ptr, output_ptr,
                      q_row_stride, k_row_stride, v_row_stride, output_row_stride,
                      scale,
                      SEQUENCE_LENGTH: tl.constexpr, HEAD_DIM: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    dimensions = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + rows[:, None] * q_row_stride + dimensions[None, :])

    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    for start in range(0, SEQUENCE_LENGTH, BLOCK_N):
        keys = start + tl.arange(0, BLOCK_N)
        k = tl.load(k_ptr + keys[:, None] * k_row_stride + dimensions[None, :])
        v = tl.load(v_ptr + keys[:, None] * v_row_stride + dimensions[None, :])
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale

        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        alpha = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        accumulator = accumulator * alpha[:, None] + tl.dot(
            probabilities, v, input_precision="ieee"
        )
        running_max = new_max

    output = accumulator / running_sum[:, None]
    tl.store(output_ptr + rows[:, None] * output_row_stride + dimensions[None, :], output)


@torch.no_grad()
def triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute aligned, single-sequence, non-causal CUDA FP32 attention."""
    if q.ndim != 2 or q.shape != k.shape or q.shape != v.shape or 0 in q.shape:
        raise ValueError("Q, K, and V must have the same nonempty (N, D) shape.")
    if any(t.device != q.device or t.device.type != "cuda" or t.dtype != torch.float32
           for t in (q, k, v)):
        raise ValueError("This milestone requires CUDA FP32 Q, K, and V on one device.")
    if not all(t.is_contiguous() for t in (q, k, v)):
        raise ValueError("Q, K, and V must be contiguous.")

    sequence_length, head_dim = q.shape
    # shortcut: milestone 07 supports aligned N and D in {16, 32, 64}; generalize in milestone 08.
    if sequence_length % 16 or head_dim not in (16, 32, 64):
        raise ValueError("N must be a multiple of 16 and D must be 16, 32, or 64.")

    output = torch.empty_like(q)
    block_m = block_n = 16
    _attention_kernel[(sequence_length // block_m,)](
        q, k, v, output,
        q.stride(0), k.stride(0), v.stride(0), output.stride(0),
        1.0 / math.sqrt(head_dim),
        SEQUENCE_LENGTH=sequence_length, HEAD_DIM=head_dim,
        BLOCK_M=block_m, BLOCK_N=block_n,
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=positive_int, default=64)
    parser.add_argument("--head-dim", type=positive_int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", choices=["random", "uniform", "rising"], default="random")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA is unavailable.")
        return 1

    torch.manual_seed(args.seed)
    tensors = [
        torch.randn(args.seq_len, args.head_dim, device="cuda", dtype=torch.float32)
        for _ in range(3)
    ]
    q, k, v = tensors
    if args.scenario == "uniform":
        q.zero_()
    elif args.scenario == "rising":
        q.zero_()
        q[:, 0] = 1.0
        k.zero_()
        k[:, 0] = torch.linspace(0, 20, args.seq_len, device="cuda") * math.sqrt(args.head_dim)

    output = triton_attention(q, k, v)
    naive = naive_attention(q[None, None], k[None, None], v[None, None])[0, 0]
    sdpa = functional.scaled_dot_product_attention(
        q[None, None], k[None, None], v[None, None], dropout_p=0.0,
    )[0, 0]
    references = {"naive attention": naive, "SDPA": sdpa}
    if args.scenario == "uniform":
        references["mean(V)"] = v.mean(dim=0).expand_as(v)
    torch.cuda.synchronize()

    passed = bool(torch.isfinite(output).all())
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch {torch.__version__}; Triton {triton.__version__}; "
          f"N={args.seq_len}; D={args.head_dim}; scenario={args.scenario}; seed={args.seed}")
    print(f"Output shape: {tuple(output.shape)}")
    for name, reference in references.items():
        error = (output - reference).abs().max().item()
        close = torch.allclose(output, reference, rtol=1e-4, atol=1e-5)
        passed = passed and close
        print(f"{'PASS' if close else 'FAIL'} vs {name}: maximum absolute error={error:.8e}")
    print(f"{'PASS' if passed else 'FAIL'}: finite output and reference agreement "
          "(rtol=1e-4, atol=1e-5)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
