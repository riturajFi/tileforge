"""Run a CPU attention memory baseline in a fresh process per sequence length."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import torch
import torch.nn.functional as functional

from tileforge.naive import naive_attention, positive_int
from tileforge.tiled import tiled_attention


def read_system_file(path: str) -> str | None:
    location = Path(path)
    return location.read_text().strip() if location.exists() else None


def peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB. Restrict this baseline to Linux below.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def current_rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS was not reported by /proc/self/status")


@torch.no_grad()
def measure(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    n = args.worker_seq_len
    shape = (1, 1, n, args.head_dim)
    generator = torch.Generator().manual_seed(args.seed)
    q, k, v = [torch.randn(shape, generator=generator) for _ in range(3)]

    def attention():
        if args.implementation == "naive":
            return naive_attention(q, k, v)
        return tiled_attention(q[0, 0], k[0, 0], v[0, 0],
                               args.query_tile, args.kv_tile)[None, None]

    # Measure the first actual call, before warmup, SDPA, or profiling can raise
    # the process high-water mark. Q/K/V already exist at this baseline.
    rss_before = current_rss_bytes()
    hwm_before = peak_rss_bytes()
    output = attention()
    hwm_after = peak_rss_bytes()
    rss_after = current_rss_bytes()

    reference = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    max_error = (output - reference).abs().max().item()
    if not torch.allclose(output, reference, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"N={n}: SDPA comparison failed; max error {max_error}")
    del output, reference

    for _ in range(args.warmup):
        output = attention()
        del output
    samples_ms = []
    for _ in range(args.repeats):
        start = time.perf_counter_ns()
        output = attention()
        elapsed = time.perf_counter_ns() - start
        samples_ms.append(elapsed / 1e6)
        del output

    # Profile a separate call after timing. Positive self-memory values are
    # allocations attributed to operators, NOT process peak memory.
    allocations = None
    if not args.skip_profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True,
        ) as profile:
            output = attention()
        allocations = [
            {"operator": event.key, "calls": event.count,
             "self_cpu_memory_bytes": event.self_cpu_memory_usage}
            for event in profile.key_averages() if event.self_cpu_memory_usage > 0
        ]
    return {
        "implementation": args.implementation,
        "sequence_length": n,
        "shape": list(shape),
        "qkv_bytes": sum(t.numel() * t.element_size() for t in (q, k, v)),
        "one_score_matrix_bytes": n * n * q.element_size(),
        "three_live_score_matrices_bytes": (
            3 * n * n * q.element_size() if args.implementation == "naive" else None),
        "output_bytes": q.numel() * q.element_size(),
        "rss_before_first_call_bytes": rss_before,
        "rss_after_first_call_bytes": rss_after,
        "hwm_before_first_call_bytes": hwm_before,
        "hwm_after_first_call_bytes": hwm_after,
        "hwm_increase_bytes": hwm_after - hwm_before,
        "latency_samples_ms": samples_ms,
        "median_latency_ms": statistics.median(samples_ms),
        "max_absolute_error_vs_sdpa": max_error,
        "matches_sdpa": True,
        "profile_positive_self_allocations": allocations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=positive_int,
                        default=[128, 256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--head-dim", type=positive_int, default=64)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--warmup", type=positive_int, default=2)
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--implementation", choices=["naive", "tiled"], default="naive")
    parser.add_argument("--query-tile", type=positive_int, default=64)
    parser.add_argument("--kv-tile", type=positive_int, default=128)
    parser.add_argument("--skip-profile", action="store_true",
                        help="Skip operator profiling, keeping first-call RSS and timing measurements.")
    parser.add_argument("--output", type=Path, help="New JSON results file; refuses to overwrite.")
    parser.add_argument("--worker-seq-len", type=positive_int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("RSS measurements currently require Linux /proc and ru_maxrss semantics.")
    if args.worker_seq_len is not None:
        print(json.dumps(measure(args)))
        return 0
    if args.output and args.output.exists():
        parser.error(f"Results already exist: {args.output}; choose a new filename.")

    metadata = {
        "started_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "command": [sys.executable, "-m", "tileforge.memory_baseline", *sys.argv[1:]],
        "python": platform.python_version(), "pytorch": torch.__version__,
        "platform": platform.platform(), "device": "cpu", "dtype": "float32",
        "cuda_available": torch.cuda.is_available(),
        "cpu_model": next((line.split(":", 1)[1].strip()
                           for line in Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")), "unknown"),
        "logical_cpus_visible": os.cpu_count(),
        "cgroup_cpu_max": read_system_file("/sys/fs/cgroup/cpu.max"),
        "cgroup_memory_max": read_system_file("/sys/fs/cgroup/memory.max"),
        "batch": 1, "heads": 1, "head_dim": args.head_dim,
        "threads": args.threads, "interop_threads": 1, "seed": args.seed,
        "warmup": args.warmup, "repeats": args.repeats,
        "grad_enabled": False, "causal": False, "dropout": 0.0,
        "correctness_rtol": 1e-5, "correctness_atol": 1e-6,
        "implementation": args.implementation,
        "query_tile": args.query_tile if args.implementation == "tiled" else None,
        "kv_tile": args.kv_tile if args.implementation == "tiled" else None,
        "operator_profiling": not args.skip_profile,
    }
    rows = []
    print(f"Implementation: {args.implementation}", flush=True)
    print("N | full score MiB | peak RSS MiB | HWM increase MiB | median ms", flush=True)
    for n in args.lengths:
        command = [sys.executable, "-m", "tileforge.memory_baseline",
                   "--worker-seq-len", str(n), "--head-dim", str(args.head_dim),
                   "--threads", str(args.threads), "--warmup", str(args.warmup),
                   "--repeats", str(args.repeats), "--seed", str(args.seed),
                   "--implementation", args.implementation,
                   "--query-tile", str(args.query_tile), "--kv-tile", str(args.kv_tile)]
        if args.skip_profile:
            command.append("--skip-profile")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"N={n}: worker exited {result.returncode}\n{result.stderr}")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        row = json.loads(result.stdout)
        rows.append(row)
        mib = 1024 ** 2
        print(f"{n} | {row['one_score_matrix_bytes'] / mib:.3f} | "
              f"{row['hwm_after_first_call_bytes'] / mib:.2f} | "
              f"{row['hwm_increase_bytes'] / mib:.2f} | "
              f"{row['median_latency_ms']:.3f}", flush=True)
    report = {"metadata": metadata, "results": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
