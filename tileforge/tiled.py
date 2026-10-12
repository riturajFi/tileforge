"""CPU tiled attention for a full sequence of query rows."""

import argparse
import math

import torch
import torch.nn.functional as functional

from tileforge.naive import naive_attention, positive_int


@torch.no_grad()
def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_tile_size: int = 64,
    kv_tile_size: int = 128,
) -> torch.Tensor:
    """Forward-only, non-causal CPU FP32 attention for Q/K/V of shape (N, D).

    Inputs and scaled scores must be finite. Scratch space is O(Br*Bc + Br*D),
    excluding inputs and the (N, D) output. Oversized tiles are clipped; if both
    cover the entire sequence, K/V are split to avoid a full N-by-N score tile.
    N=1 returns V directly. This reference has no batching, masks, or dropout.
    """
    for size in (query_tile_size, kv_tile_size):
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("Tile sizes must be positive integers.")
    if q.ndim != 2 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q, K, and V must have the same (N, D) shape.")
    if any(size == 0 for size in q.shape):
        raise ValueError("Sequence length and head dimension must be nonzero.")
    if any(t.device.type != "cpu" or t.dtype != torch.float32 for t in (q, k, v)):
        raise ValueError("This milestone requires CPU float32 inputs.")

    n, d = q.shape
    if n == 1:
        return v.clone()
    query_step = min(query_tile_size, n)
    kv_step = min(kv_tile_size, n)
    if query_step == n and kv_step == n:
        kv_step = max(1, n // 2)
    output = torch.empty_like(q)
    scale = 1.0 / math.sqrt(d)

    for query_start in range(0, n, query_step):
        query_stop = min(query_start + query_step, n)
        q_tile = q[query_start:query_stop]
        rows = query_stop - query_start
        m = q.new_full((rows, 1), -float("inf"))
        l = q.new_zeros((rows, 1))
        acc = q.new_zeros((rows, d))
        for kv_start in range(0, n, kv_step):
            kv_stop = min(kv_start + kv_step, n)
            scores = (q_tile @ k[kv_start:kv_stop].T) * scale
            new_m = torch.maximum(m, scores.amax(dim=1, keepdim=True))
            alpha = torch.exp(m - new_m)
            p = torch.exp(scores - new_m)
            l = l * alpha + p.sum(dim=1, keepdim=True)
            acc = acc * alpha + p @ v[kv_start:kv_stop]
            m = new_m
        output[query_start:query_stop] = acc / l
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=positive_int, default=17)
    parser.add_argument("--head-dim", type=positive_int, default=8)
    parser.add_argument("--query-tile", type=positive_int, default=4)
    parser.add_argument("--kv-tile", type=positive_int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", choices=["random", "uniform", "extreme"], default="random")
    parser.add_argument("--strided", action="store_true", help="Use non-contiguous Q/K/V views.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)
    width = args.head_dim * (2 if args.strided else 1)
    tensors = [torch.randn(args.seq_len, width, generator=generator) for _ in range(3)]
    q, k, v = [t[:, ::2] if args.strided else t for t in tensors]
    if args.scenario == "uniform":
        q.zero_()
    elif args.scenario == "extreme":
        q.zero_()
        q[:, 0] = torch.linspace(-1, 1, args.seq_len)
        k.zero_()
        k[:, 0] = torch.linspace(1000, 1020, args.seq_len) * math.sqrt(args.head_dim)

    with torch.no_grad():
        output = tiled_attention(q, k, v, args.query_tile, args.kv_tile)
        reference = naive_attention(q[None, None], k[None, None], v[None, None])[0, 0]
        sdpa = functional.scaled_dot_product_attention(
            q[None, None], k[None, None], v[None, None], dropout_p=0.0,
        )[0, 0]
        references = {"naive attention": reference, "SDPA": sdpa}
        if args.scenario == "uniform":
            references["mean(V)"] = v.mean(dim=0).expand_as(v)
        matches = bool(torch.isfinite(output).all())
        print(f"N={args.seq_len}, D={args.head_dim}, query_tile={args.query_tile}, "
              f"kv_tile={args.kv_tile}, scenario={args.scenario}, seed={args.seed}, "
              f"strided={args.strided}; CPU FP32")
        print(f"Output shape: {tuple(output.shape)}; first row: {output[0]}")
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
