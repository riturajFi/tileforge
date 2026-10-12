"""Naive scaled dot-product attention and a runnable CPU comparison with SDPA."""

import argparse
import math

import torch
import torch.nn.functional as functional


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute non-causal self-attention without masking or dropout.

    Inputs have identical (batch, heads, sequence, head_dim) shapes, floating
    dtype, and device. The output has the same shape. This baseline deliberately
    materializes the full (batch, heads, sequence, sequence) score matrix.
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q, K, and V must have the same (B, H, N, D) shape.")
    if any(size == 0 for size in q.shape):
        raise ValueError("Q, K, and V dimensions must be nonzero.")
    if not (q.dtype == k.dtype == v.dtype and q.is_floating_point()):
        raise ValueError("Q, K, and V must share a floating-point dtype.")
    if not q.device == k.device == v.device:
        raise ValueError("Q, K, and V must be on the same device.")

    scores = q @ k.transpose(-2, -1)
    scaled_scores = scores / math.sqrt(q.shape[-1])
    weights = torch.softmax(scaled_scores, dim=-1)
    return weights @ v


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-len", type=positive_int, default=4)
    parser.add_argument("--head-dim", type=positive_int, default=4)
    args = parser.parse_args()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    shape = (1, 1, args.seq_len, args.head_dim)
    q, k, v = [
        torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")
        for _ in range(3)
    ]
    with torch.no_grad():
        output = naive_attention(q, k, v)
        reference = functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False,
        )

    max_error = (output - reference).abs().max().item()
    matches = torch.allclose(output, reference, rtol=1e-5, atol=1e-6)
    print(f"Shape (B, H, N, D): {shape}; device: cpu; dtype: float32; seed: {args.seed}")
    print(f"Scale: 1 / sqrt({args.head_dim}) = {1 / math.sqrt(args.head_dim):.6f}")
    print(f"Naive output:\n{output}")
    print(f"SDPA output:\n{reference}")
    print(f"Maximum absolute error: {max_error:.8e}")
    print(f"{'PASS' if matches else 'FAIL'}: matches SDPA (rtol=1e-5, atol=1e-6)")
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
