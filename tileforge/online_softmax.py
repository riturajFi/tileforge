"""One-query CPU attention using a streaming, numerically stable softmax."""

import argparse
import math

import torch
import torch.nn.functional as functional

from tileforge.naive import positive_int


@torch.no_grad()
def online_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 64,
    *,
    trace: bool = False,
) -> torch.Tensor:
    """Attend one CPU FP32 query (D,) to keys and values (N, D).

    Forward only, without masking or dropout. Inputs and scaled dot products
    must be finite. Scratch space is O(chunk_size + D), excluding inputs and
    output; only one chunk of scores is materialized at a time. Autograd is
    disabled so it cannot retain intermediates from earlier chunks.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if q.ndim != 1 or k.ndim != 2 or v.shape != k.shape or k.shape[1] != q.shape[0]:
        raise ValueError("Expected Q with shape (D,) and K/V with the same shape (N, D).")
    if q.numel() == 0 or k.shape[0] == 0:
        raise ValueError("Sequence length and head dimension must be nonzero.")
    if any(t.device.type != "cpu" or t.dtype != torch.float32 for t in (q, k, v)):
        raise ValueError("This milestone requires CPU float32 inputs.")

    scale = 1.0 / math.sqrt(q.shape[0])
    m = q.new_tensor(-float("inf"))
    l = q.new_zeros(())
    acc = torch.zeros_like(q)

    for start in range(0, k.shape[0], chunk_size):
        stop = min(start + chunk_size, k.shape[0])
        scores = (k[start:stop] @ q) * scale
        new_m = torch.maximum(m, scores.max())
        # Express previous accumulators relative to the new maximum.
        alpha = torch.exp(m - new_m)
        p = torch.exp(scores - new_m)
        l = l * alpha + p.sum()
        acc = acc * alpha + p @ v[start:stop]
        m = new_m
        if trace:
            print(f"chunk [{start}:{stop}]: m={m.item():.6f}, "
                  f"alpha={alpha.item():.6e}, l={l.item():.6f}")

    return acc / l


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=positive_int, default=4)
    parser.add_argument("--head-dim", type=positive_int, default=4)
    parser.add_argument("--chunk-size", type=positive_int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", choices=["random", "uniform", "rising", "negative"],
                        default="random")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)
    q = torch.randn(args.head_dim, generator=generator)
    k = torch.randn(args.seq_len, args.head_dim, generator=generator)
    v = torch.randn(args.seq_len, args.head_dim, generator=generator)
    if args.scenario == "uniform":
        q.zero_()
    elif args.scenario in ("rising", "negative"):
        q.zero_()
        q[0] = 1.0
        k.zero_()
        offset = 1000.0 if args.scenario == "rising" else -1000.0
        k[:, 0] = torch.linspace(offset, offset + 20, args.seq_len) * math.sqrt(args.head_dim)

    with torch.no_grad():
        output = online_attention(q, k, v, args.chunk_size, trace=args.trace)
        full_scores = (k @ q) / math.sqrt(args.head_dim)
        reference = torch.softmax(full_scores, dim=0) @ v
        sdpa = functional.scaled_dot_product_attention(
            q[None, None, None, :], k[None, None, :, :], v[None, None, :, :],
            dropout_p=0.0, is_causal=False,
        ).reshape(-1)
        references = {"full softmax": reference, "SDPA": sdpa}
        if args.scenario == "uniform":
            references["mean(V)"] = v.mean(dim=0)
        matches = bool(torch.isfinite(output).all())
        print(f"N={args.seq_len}, D={args.head_dim}, chunk_size={args.chunk_size}, "
              f"scenario={args.scenario}, seed={args.seed}; CPU FP32")
        print(f"Score range: [{full_scores.min().item():.6f}, {full_scores.max().item():.6f}]")
        print(f"Online output:\n{output}")
        for name, expected in references.items():
            error = (output - expected).abs().max().item()
            close = torch.allclose(output, expected, rtol=1e-5, atol=1e-6)
            matches = matches and close
            print(f"{'PASS' if close else 'FAIL'} vs {name}: max absolute error={error:.8e}")
        print(f"{'PASS' if matches else 'FAIL'}: finite output and reference agreement "
              "(rtol=1e-5, atol=1e-6)")
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
