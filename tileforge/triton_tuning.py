"""Correctness-gated tuning for the fused Triton attention kernel."""

import argparse
import statistics

import torch
import torch.nn.functional as functional

from tileforge.naive import positive_int
from tileforge.triton_attention import _triton_attention_with_config


Config = tuple[int, int, int, int]


def _measure(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: Config,
    warmups: int,
    repetitions: int,
) -> tuple[float, float]:
    output = _triton_attention_with_config(q, k, v, config=config)
    reference = functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    error = (output.float() - reference.float()).abs().max().item()
    if not torch.isfinite(output).all() or not torch.allclose(
        output.float(), reference.float(), rtol=2e-2, atol=2e-3
    ):
        raise RuntimeError(f"configuration {config} failed correctness; error={error:.8e}")

    for _ in range(warmups):
        _triton_attention_with_config(q, k, v, config=config)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _triton_attention_with_config(q, k, v, config=config)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), error


def _choose(
    name: str,
    configs: list[Config],
    inputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    warmups: int,
    repetitions: int,
) -> Config:
    results = []
    print(f"\n{name}")
    for config in configs:
        latencies = []
        for sequence_length, (q, k, v) in inputs.items():
            latency, error = _measure(q, k, v, config, warmups, repetitions)
            latencies.append(latency)
            print(f"config={config} N={sequence_length} median_ms={latency:.6f} "
                  f"max_error={error:.8e}")
        score = statistics.geometric_mean(latencies)
        results.append((score, config))
        print(f"config={config} geometric_mean_ms={score:.6f}")
    score, winner = min(results)
    print(f"selected_{name}={winner} geometric_mean_ms={score:.6f}")
    return winner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=positive_int, default=10)
    parser.add_argument("--repetitions", type=positive_int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA is unavailable.")
        return 1

    torch.manual_seed(0)
    inputs = {
        sequence_length: tuple(
            torch.randn((1, 8, sequence_length, 64), device="cuda", dtype=torch.float16)
            for _ in range(3)
        )
        for sequence_length in (128, 256, 512, 1024)
    }
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("workload: B=1 H=8 D=64 dtype=float16 causal=False; "
          f"warmups={args.warmups}; repetitions={args.repetitions}")

    tile_configs = [
        (16, 16, 4, 2),
        (16, 32, 4, 2),
        (32, 16, 4, 2),
        (32, 32, 4, 2),
        (64, 32, 4, 2),
        (64, 64, 4, 2),
    ]
    selected = _choose("tiles", tile_configs, inputs, args.warmups, args.repetitions)
    block_m, block_n, _, num_stages = selected
    warp_configs = [
        (block_m, block_n, num_warps, num_stages) for num_warps in (2, 4, 8)
    ]
    selected = _choose("warps", warp_configs, inputs, args.warmups, args.repetitions)
    block_m, block_n, num_warps, _ = selected
    stage_configs = [
        (block_m, block_n, num_warps, num_stages) for num_stages in (2, 3, 4)
    ]
    selected = _choose("stages", stage_configs, inputs, args.warmups, args.repetitions)
    print(f"\nTUNED_CONFIG={selected}")
    print("PASS: every measured configuration matched PyTorch SDPA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
