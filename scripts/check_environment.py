"""Report capabilities and exercise PyTorch; missing optional GPU tools are OK."""

import argparse
import importlib.metadata
import platform
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="Fail unless CUDA execution works.")
    parser.add_argument("--require-nvcc", action="store_true", help="Fail unless the CUDA compiler runs.")
    args = parser.parse_args()

    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        print(f"FAIL: PyTorch could not be imported: {exc}", file=sys.stderr)
        return 1

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda or 'none (CPU build)'}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA usable by PyTorch: {cuda_available}")
    try:
        print(f"Triton installed version: {importlib.metadata.version('triton')} (kernel execution not tested)")
    except importlib.metadata.PackageNotFoundError:
        print("Triton: not installed (needed at milestone 05)")

    # nvidia-smi can detect hardware even when this environment uses CPU PyTorch.
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        print(f"NVIDIA driver query (exit {result.returncode}): {(result.stdout or result.stderr).strip()}")
    else:
        print("NVIDIA driver query: nvidia-smi unavailable")

    nvcc = shutil.which("nvcc")
    nvcc_ok = False
    if nvcc:
        result = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=30)
        nvcc_ok = result.returncode == 0
        print(f"CUDA compiler (exit {result.returncode}): {(result.stdout or result.stderr).strip()}")
    else:
        print("CUDA compiler: nvcc unavailable (needed at milestone 10)")

    devices = ["cpu"]
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            device = f"cuda:{index}"
            print(f"{device}: {torch.cuda.get_device_name(index)}, capability {torch.cuda.get_device_capability(index)}")
            devices.append(device)

    for device in devices:
        # Zero scores give uniform weights; SDPA must return the mean of V.
        q = torch.zeros((1, 1, 4, 4), device=device)
        k = torch.zeros_like(q)
        v = torch.arange(16, dtype=torch.float32, device=device).reshape(1, 1, 4, 4)
        torch.testing.assert_close(v @ torch.eye(4, device=device), v)
        output = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        expected = v.mean(dim=-2, keepdim=True).expand_as(v)
        torch.testing.assert_close(output, expected)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        print(f"PASS: {device} tensor allocation, matrix multiplication, and SDPA smoke check")

    if args.require_cuda and not cuda_available:
        print("FAIL: CUDA execution was required but is unavailable.", file=sys.stderr)
        return 1
    if args.require_nvcc and not nvcc_ok:
        print("FAIL: a working nvcc was required but is unavailable.", file=sys.stderr)
        return 1
    print("PASS: environment check (optional missing GPU capabilities reported above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
