"""Fused Triton attention forward kernel with online softmax."""

import argparse
import math

import torch
import torch.nn.functional as functional
import triton
import triton.language as tl

from tileforge.naive import positive_int


_TUNED_CONFIG = (64, 64, 4, 2)


@triton.jit
def _attention_kernel(q_ptr, k_ptr, v_ptr, output_ptr,
                      q_head_stride, k_head_stride, v_head_stride, output_head_stride,
                      q_row_stride, k_row_stride, v_row_stride, output_row_stride,
                      scale,
                      SEQUENCE_LENGTH: tl.constexpr, HEAD_DIM: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_D: tl.constexpr, CAUSAL: tl.constexpr):
    batch_head = tl.program_id(1)
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    dimensions = tl.arange(0, BLOCK_D)
    q = tl.load(
        q_ptr + batch_head * q_head_stride
        + rows[:, None] * q_row_stride + dimensions[None, :],
        mask=(rows[:, None] < SEQUENCE_LENGTH) & (dimensions[None, :] < HEAD_DIM),
        other=0.0,
    )

    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start in range(0, SEQUENCE_LENGTH, BLOCK_N):
        keys = start + tl.arange(0, BLOCK_N)
        key_mask = (keys[:, None] < SEQUENCE_LENGTH) & (dimensions[None, :] < HEAD_DIM)
        k = tl.load(
            k_ptr + batch_head * k_head_stride
            + keys[:, None] * k_row_stride + dimensions[None, :],
            mask=key_mask,
            other=0.0,
        )
        v = tl.load(
            v_ptr + batch_head * v_head_stride
            + keys[:, None] * v_row_stride + dimensions[None, :],
            mask=key_mask,
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        score_mask = keys[None, :] < SEQUENCE_LENGTH
        if CAUSAL:
            score_mask &= keys[None, :] <= rows[:, None]
        scores = tl.where(score_mask, scores, -float("inf"))

        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        alpha = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        accumulator = accumulator * alpha[:, None] + tl.dot(
            probabilities.to(v.dtype), v, input_precision="ieee"
        )
        running_max = new_max

    output = accumulator / running_sum[:, None]
    tl.store(
        output_ptr + batch_head * output_head_stride
        + rows[:, None] * output_row_stride + dimensions[None, :],
        output,
        mask=(rows[:, None] < SEQUENCE_LENGTH) & (dimensions[None, :] < HEAD_DIM),
    )


@torch.no_grad()
def _triton_attention_with_config(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    config: tuple[int, int, int, int],
) -> torch.Tensor:
    if q.ndim not in (2, 4) or q.shape != k.shape or q.shape != v.shape or 0 in q.shape:
        raise ValueError("Q, K, and V must share a nonempty (N, D) or (B, H, N, D) shape.")
    if any(t.device != q.device or t.device.type != "cuda" or t.dtype != q.dtype
           for t in (q, k, v)) or q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Q, K, and V must share one CUDA device and use FP32, FP16, or BF16.")
    if not all(t.is_contiguous() for t in (q, k, v)):
        raise ValueError("Q, K, and V must be contiguous.")
    if not isinstance(causal, bool):
        raise ValueError("causal must be a boolean.")

    sequence_length, head_dim = q.shape[-2:]
    # shortcut: D above 128 needs another tiling strategy; add one if later benchmarks need it.
    if head_dim > 128:
        raise ValueError("Head dimensions above 128 are not supported.")

    original_shape = q.shape
    q_flat, k_flat, v_flat = [t.reshape(-1, sequence_length, head_dim) for t in (q, k, v)]
    output = torch.empty_like(q_flat)
    block_m, block_n, num_warps, num_stages = config
    block_d = max(16, triton.next_power_of_2(head_dim))
    grid = (triton.cdiv(sequence_length, block_m), q_flat.shape[0])
    _attention_kernel[grid](
        q_flat, k_flat, v_flat, output,
        q_flat.stride(0), k_flat.stride(0), v_flat.stride(0), output.stride(0),
        q_flat.stride(1), k_flat.stride(1), v_flat.stride(1), output.stride(1),
        1.0 / math.sqrt(head_dim),
        SEQUENCE_LENGTH=sequence_length, HEAD_DIM=head_dim,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d, CAUSAL=causal,
        num_warps=num_warps, num_stages=num_stages,
    )
    return output.reshape(original_shape)


def triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """Compute forward-only CUDA attention for `(N, D)` or `(B, H, N, D)`."""
    return _triton_attention_with_config(q, k, v, causal=causal, config=_TUNED_CONFIG)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--heads", type=positive_int, default=1)
    parser.add_argument("--seq-len", type=positive_int, default=65)
    parser.add_argument("--head-dim", type=positive_int, default=40)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", choices=["random", "uniform", "rising"], default="random")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA is unavailable.")
        return 1

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    torch.manual_seed(args.seed)
    shape = (args.batch, args.heads, args.seq_len, args.head_dim)
    q, k, v = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3)]
    if args.scenario == "uniform":
        q.zero_()
    elif args.scenario == "rising":
        q.zero_()
        q[..., 0] = 1.0
        k.zero_()
        k[..., 0] = torch.linspace(0, 20, args.seq_len, device="cuda", dtype=dtype) \
                    * math.sqrt(args.head_dim)

    output = triton_attention(q, k, v, causal=args.causal)
    scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(args.head_dim)
    if args.causal:
        mask = torch.ones(args.seq_len, args.seq_len, device="cuda", dtype=torch.bool).tril()
        scores.masked_fill_(~mask, -float("inf"))
    fp32_reference = torch.softmax(scores, dim=-1) @ v.float()
    sdpa = functional.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=args.causal,
    )
    references = {"FP32 reference": fp32_reference, "SDPA": sdpa.float()}
    if args.scenario == "uniform" and not args.causal:
        references["mean(V)"] = v.float().mean(dim=-2, keepdim=True).expand_as(fp32_reference)
    torch.cuda.synchronize()

    tolerances = {
        torch.float32: (1e-4, 1e-5),
        torch.float16: (2e-2, 2e-3),
        torch.bfloat16: (3e-2, 1e-2),
    }
    rtol, atol = tolerances[dtype]
    actual = output.float()
    passed = bool(torch.isfinite(actual).all())
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch {torch.__version__}; Triton {triton.__version__}; shape={shape}; "
          f"dtype={args.dtype}; causal={args.causal}; scenario={args.scenario}; seed={args.seed}")
    for name, reference in references.items():
        error = (actual - reference).abs().max().item()
        close = torch.allclose(actual, reference, rtol=rtol, atol=atol)
        passed = passed and close
        print(f"{'PASS' if close else 'FAIL'} vs {name}: maximum absolute error={error:.8e}")
    print(f"{'PASS' if passed else 'FAIL'}: finite output and reference agreement "
          f"(rtol={rtol:g}, atol={atol:g})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
