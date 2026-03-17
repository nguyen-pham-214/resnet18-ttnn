# test_utility.py

import time
from pathlib import Path

import pandas as pd
import torch

from model_torch import create_torch_model
from model_ttnn import create_ttnn_model, run_ttnn_model

try:
    from model_ttnn import synchronize_device
except ImportError:
    def synchronize_device():
        pass


# ----------------------------
# Configuration
# ----------------------------
DEFAULT_SEED = 0
DEFAULT_INPUT_SHAPE = (1, 3, 32, 32)
DEFAULT_DEVICE = "cpu"
DEFAULT_WARMUP = 3
DEFAULT_REPEAT = 10
DEFAULT_PCC_THRESHOLD = 0.99
DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2
DEFAULT_OUTPUT_DIR = "results"


# ----------------------------
# Data generation
# ----------------------------
def generate_input(shape=DEFAULT_INPUT_SHAPE, seed=DEFAULT_SEED, device=DEFAULT_DEVICE):
    torch.manual_seed(seed)
    x = torch.randn(*shape, dtype=torch.float32)
    return x.to(device)


# ----------------------------
# Metrics
# ----------------------------
def compute_pcc(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.detach().float().cpu().reshape(-1)
    b = b.detach().float().cpu().reshape(-1)

    a_centered = a - a.mean()
    b_centered = b - b.mean()

    denom = torch.sqrt(torch.sum(a_centered ** 2)) * torch.sqrt(torch.sum(b_centered ** 2))
    denom = torch.clamp(denom, min=eps)

    pcc = torch.sum(a_centered * b_centered) / denom
    return float(pcc.item())


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float().cpu() - b.detach().float().cpu()).abs().max().item())


def mean_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float().cpu() - b.detach().float().cpu()).abs().mean().item())


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = a.detach().float().cpu() - b.detach().float().cpu()
    return float(torch.sqrt(torch.mean(diff ** 2)).item())


def allclose_result(a: torch.Tensor, b: torch.Tensor, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL) -> bool:
    return bool(torch.allclose(a.detach().cpu(), b.detach().cpu(), atol=atol, rtol=rtol))


# ----------------------------
# Runtime helpers
# ----------------------------
def synchronize_torch(device: str):
    if "cuda" in str(device).lower() and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_runtime(fn, *args, warmup=DEFAULT_WARMUP, repeat=DEFAULT_REPEAT, sync_fn=None, **kwargs):
    if sync_fn is None:
        sync_fn = lambda: None

    # warmup
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
        sync_fn()

    times_ms = []
    last_output = None

    for _ in range(repeat):
        start = time.perf_counter()
        last_output = fn(*args, **kwargs)
        sync_fn()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    stats = {
        "mean_ms": sum(times_ms) / len(times_ms),
        "std_ms": pd.Series(times_ms).std(ddof=0),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "runs": len(times_ms),
    }
    return last_output, times_ms, stats


# ----------------------------
# Model runners
# ----------------------------
@torch.no_grad()
def run_torch_model(torch_model, x):
    return torch_model(x)


@torch.no_grad()
def compare_torch_vs_ttnn(
    input_shape=DEFAULT_INPUT_SHAPE,
    device=DEFAULT_DEVICE,
    seed=DEFAULT_SEED,
    warmup=DEFAULT_WARMUP,
    repeat=DEFAULT_REPEAT,
    pcc_threshold=DEFAULT_PCC_THRESHOLD,
    atol=DEFAULT_ATOL,
    rtol=DEFAULT_RTOL,
    output_dir=DEFAULT_OUTPUT_DIR,
    excel_name="comparison.xlsx",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1) Create input
    x = generate_input(shape=input_shape, seed=seed, device=device)

    # 2) Create models
    torch_model = create_torch_model(device=device)
    ttnn_model = create_ttnn_model(torch_model=torch_model, device=device)

    # 3) Run torch
    torch_out, torch_times, torch_stats = measure_runtime(
        run_torch_model,
        torch_model,
        x,
        warmup=warmup,
        repeat=repeat,
        sync_fn=lambda: synchronize_torch(device),
    )

    # 4) Run ttnn
    ttnn_out, ttnn_times, ttnn_stats = measure_runtime(
        run_ttnn_model,
        ttnn_model,
        x.detach().cpu(),   # safer default: pass CPU torch tensor into TTNN wrapper
        warmup=warmup,
        repeat=repeat,
        sync_fn=synchronize_device,
    )

    # 5) Normalize outputs to torch tensors on CPU
    torch_out_cpu = torch_out.detach().float().cpu()
    if not isinstance(ttnn_out, torch.Tensor):
        raise TypeError("run_ttnn_model(...) must return a torch.Tensor")
    ttnn_out_cpu = ttnn_out.detach().float().cpu()

    # 6) Shape check
    same_shape = tuple(torch_out_cpu.shape) == tuple(ttnn_out_cpu.shape)

    # 7) Metrics
    metrics = {
        "same_shape": same_shape,
        "torch_shape": str(tuple(torch_out_cpu.shape)),
        "ttnn_shape": str(tuple(ttnn_out_cpu.shape)),
        "torch_dtype": str(torch_out_cpu.dtype),
        "ttnn_dtype": str(ttnn_out_cpu.dtype),
        "pcc": compute_pcc(torch_out_cpu, ttnn_out_cpu),
        "max_abs_error": max_abs_error(torch_out_cpu, ttnn_out_cpu),
        "mean_abs_error": mean_abs_error(torch_out_cpu, ttnn_out_cpu),
        "rmse": rmse(torch_out_cpu, ttnn_out_cpu),
        "allclose": allclose_result(torch_out_cpu, ttnn_out_cpu, atol=atol, rtol=rtol),
    }
    metrics["pcc_pass"] = metrics["pcc"] >= pcc_threshold

    # 8) Summary table
    runtime_df = pd.DataFrame(
        [
            {
                "model": "torch",
                "mean_ms": torch_stats["mean_ms"],
                "std_ms": torch_stats["std_ms"],
                "min_ms": torch_stats["min_ms"],
                "max_ms": torch_stats["max_ms"],
                "runs": torch_stats["runs"],
            },
            {
                "model": "ttnn",
                "mean_ms": ttnn_stats["mean_ms"],
                "std_ms": ttnn_stats["std_ms"],
                "min_ms": ttnn_stats["min_ms"],
                "max_ms": ttnn_stats["max_ms"],
                "runs": ttnn_stats["runs"],
            },
        ]
    )

    metrics_df = pd.DataFrame([metrics])

    comparison_df = pd.DataFrame(
        [
            {
                "item": "shape_match",
                "value": metrics["same_shape"],
            },
            {
                "item": "pcc",
                "value": metrics["pcc"],
            },
            {
                "item": "pcc_threshold",
                "value": pcc_threshold,
            },
            {
                "item": "pcc_pass",
                "value": metrics["pcc_pass"],
            },
            {
                "item": "allclose",
                "value": metrics["allclose"],
            },
            {
                "item": "max_abs_error",
                "value": metrics["max_abs_error"],
            },
            {
                "item": "mean_abs_error",
                "value": metrics["mean_abs_error"],
            },
            {
                "item": "rmse",
                "value": metrics["rmse"],
            },
            {
                "item": "torch_shape",
                "value": metrics["torch_shape"],
            },
            {
                "item": "ttnn_shape",
                "value": metrics["ttnn_shape"],
            },
        ]
    )

    # 9) Print tables
    print("\n" + "=" * 80)
    print("RUNTIME COMPARISON")
    print("=" * 80)
    print(runtime_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("NUMERICAL COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    # 10) Export excel
    excel_path = output_path / excel_name
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        runtime_df.to_excel(writer, sheet_name="runtime_summary", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics_summary", index=False)

        pd.DataFrame({"torch_time_ms": torch_times}).to_excel(
            writer, sheet_name="torch_raw_times", index=False
        )
        pd.DataFrame({"ttnn_time_ms": ttnn_times}).to_excel(
            writer, sheet_name="ttnn_raw_times", index=False
        )

        pd.DataFrame(
            {
                "torch_output_flat": torch_out_cpu.reshape(-1).numpy(),
                "ttnn_output_flat": ttnn_out_cpu.reshape(-1).numpy(),
            }
        ).to_excel(writer, sheet_name="output_compare", index=False)

    print(f"\nExcel exported to: {excel_path.resolve()}")

    return {
        "torch_output": torch_out_cpu,
        "ttnn_output": ttnn_out_cpu,
        "runtime_df": runtime_df,
        "metrics_df": metrics_df,
        "excel_path": str(excel_path.resolve()),
    }


if __name__ == "__main__":
    compare_torch_vs_ttnn(
        input_shape=(1, 3, 32, 32),
        device="cpu",
        seed=0,
        warmup=3,
        repeat=10,
        pcc_threshold=0.99,
        atol=1e-2,
        rtol=1e-2,
        output_dir="results",
        excel_name="resnet18_torch_vs_ttnn.xlsx",
    )