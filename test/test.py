from pathlib import Path
import time
import os
import sys

import torch
import ttnn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ttnn"))
sys.path.insert(0, os.path.join(ROOT, "reference"))

from resnet18_ttnn import load_resnet18_from_torch_checkpoint
from resnet18_torch import create_torch_model

from collections import OrderedDict

def print_shape_comparison_table(torch_shapes, ttnn_shapes):
    all_keys = list(OrderedDict.fromkeys(list(torch_shapes.keys()) + list(ttnn_shapes.keys())))

    name_width = max(len("Layer"), max(len(k) for k in all_keys))
    torch_width = max(len("PyTorch Shape"), max(len(str(torch_shapes.get(k, "-"))) for k in all_keys))
    ttnn_width = max(len("TTNN Shape"), max(len(str(ttnn_shapes.get(k, "-"))) for k in all_keys))

    header = (
        f"{'Layer':<{name_width}} | "
        f"{'PyTorch Shape':<{torch_width}} | "
        f"{'TTNN Shape':<{ttnn_width}} | Match"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for k in all_keys:
        torch_shape = torch_shapes.get(k, "-")
        ttnn_shape = ttnn_shapes.get(k, "-")
        match = torch_shape == ttnn_shape
        print(
            f"{k:<{name_width}} | "
            f"{str(torch_shape):<{torch_width}} | "
            f"{str(ttnn_shape):<{ttnn_width}} | "
            f"{'YES' if match else 'NO'}"
        )

    print(sep)

def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)

    a = a - a.mean()
    b = b - b.mean()

    denom = torch.sqrt((a * a).sum()) * torch.sqrt((b * b).sum())
    if denom.item() == 0:
        return float("nan")

    return ((a * b).sum() / denom).item()

def ttnn_act_to_torch(ref: torch.Tensor, x_ttnn) -> torch.Tensor:
    if isinstance(x_ttnn, torch.Tensor):
        x = x_ttnn.detach().cpu().float()
    else:
        x = ttnn.to_torch(x_ttnn).detach().cpu().float()

    if tuple(x.shape) == tuple(ref.shape):
        return x.contiguous()

    if x.ndim == 4 and ref.ndim == 4:
        if x.shape[0] == ref.shape[0] and x.shape[-1] == ref.shape[1]:
            x = x.permute(0, 3, 1, 2).contiguous()

    if x.numel() == ref.numel():
        x = x.reshape_as(ref).contiguous()

    return x

def compare_acts(ttnn_acts: dict, torch_acts: dict, per_sample: bool = True):
    layer_names = [
        "input",
        "stem",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "head",
    ]

    if "input" not in torch_acts:
        raise KeyError("torch_acts must contain key 'input'")

    batch_size = torch_acts["input"].shape[0]
    results = {}

    print("-" * 90)
    print(f"{'Layer':<10} | {'Torch Shape':<20} | {'TTNN->Torch Shape':<20} | {'PCC':<10}")
    print("-" * 90)

    for name in layer_names:
        if name not in torch_acts:
            print(f"{name:<10} | missing in torch_acts")
            continue

        if name not in ttnn_acts:
            print(f"{name:<10} | missing in ttnn_acts")
            continue

        ref = torch_acts[name].detach().cpu().float()

        got = ttnn_act_to_torch(ref, ttnn_acts[name])

        same_shape = tuple(ref.shape) == tuple(got.shape)
        layer_pcc = compute_pcc(ref, got) if same_shape else float("nan")

        results[name] = {
            "torch_shape": tuple(ref.shape),
            "ttnn_shape": tuple(got.shape),
            "shape_match": same_shape,
            "pcc": layer_pcc,
        }

        print(
            f"{name:<10} | "
            f"{str(tuple(ref.shape)):<20} | "
            f"{str(tuple(got.shape)):<20} | "
            f"{layer_pcc:<10.6f}"
        )

        if per_sample and same_shape and ref.shape[0] == batch_size:
            sample_pccs = []
            for i in range(batch_size):
                sample_pccs.append(compute_pcc(ref[i], got[i]))
            results[name]["per_sample_pcc"] = sample_pccs

    print("-" * 90)
    return results


def main():

    weights_path = os.path.join(ROOT, "reference", "outputs", "resnet18_weights.pth")

    NUM_ITERS = 1
    BATCH_SIZE = 1
    HEIGHT = 224
    WIDTH = 224
    PCC_THRESHOLD = 0.99

    # -------------------------
    # Create torch reference model
    # -------------------------
    torch_device = "cpu"
    torch_model = create_torch_model(torch_device)
    torch_model.eval()

    # -------------------------
    # Create TTNN model
    # -------------------------
    ttnn_device = ttnn.open_device(device_id=0, l1_small_size=32768)
    

    try:
        ttnn_model = load_resnet18_from_torch_checkpoint(
            weights_path=weights_path,
            device=ttnn_device,
            batch_size=BATCH_SIZE,
            input_height=HEIGHT,
            input_width=WIDTH,
            num_classes=10,
            dtype=ttnn.bfloat16,
            head_memory_config=None,
        )

        worst_pcc = 1.0
        worst_max_abs_diff = 0.0
        worst_mean_abs_diff = 0.0
        failed_iters = []

        for i in range(NUM_ITERS):
            print(f"\n[ITER {i+1}/{NUM_ITERS}]")

            torch.manual_seed(i)
            torch_input_nchw = torch.randn((BATCH_SIZE, 3, HEIGHT, WIDTH), dtype=torch.float32)
            torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).contiguous().to(torch.bfloat16)

            # Torch forward
            with torch.no_grad():
                torch_output, torch_acts, torch_shapes = torch_model(torch_input_nchw)


            # TTNN forward
            shard_config = ttnn.create_sharded_memory_config(  
                shape=(BATCH_SIZE, 224, 224, 3),  
                core_grid=ttnn.CoreGrid(x=2, y=2),  
                strategy=ttnn.ShardStrategy.HEIGHT,  
                orientation=ttnn.ShardOrientation.ROW_MAJOR,  
                use_height_and_width_as_shard_shape=False,  
            )

            ttnn_input = ttnn.from_torch(
                torch_input_nhwc,
                device=ttnn_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=shard_config,
            )


            ttnn_output, ttnn_acts, ttnn_shapes = ttnn_model.forward(ttnn_input)
            ttnn.synchronize_device(ttnn_device)

            ttnn_output_torch = ttnn.to_torch(ttnn_output).float()

            # Normalize shapes
            torch_output = torch_output.reshape(BATCH_SIZE, -1).float()
            ttnn_output_torch = ttnn_output_torch.reshape(BATCH_SIZE, -1).float()

            pcc = compute_pcc(torch_output, ttnn_output_torch)
            max_abs_diff = torch.max(torch.abs(torch_output - ttnn_output_torch)).item()
            mean_abs_diff = torch.mean(torch.abs(torch_output - ttnn_output_torch)).item()

            print("torch output shape:", tuple(torch_output.shape))
            print("ttnn output shape:", tuple(ttnn_output_torch.shape))
            print("torch output:", tuple(torch_output))
            print("ttnn output:", tuple(ttnn_output_torch))
            print("PCC =", pcc)
            print("Max abs diff =", max_abs_diff)
            print("Mean abs diff =", mean_abs_diff)

            worst_pcc = min(worst_pcc, pcc)
            worst_max_abs_diff = max(worst_max_abs_diff, max_abs_diff)
            worst_mean_abs_diff = max(worst_mean_abs_diff, mean_abs_diff)

            if pcc <= PCC_THRESHOLD:
                failed_iters.append(
                    {
                        "iter": i + 1,
                        "pcc": pcc,
                        "max_abs_diff": max_abs_diff,
                        "mean_abs_diff": mean_abs_diff,
                    }
                )

        print("\n[SUMMARY]")
        print("Batch size =", BATCH_SIZE)
        print("Num iterations =", NUM_ITERS)
        print("Worst PCC =", worst_pcc)
        print("Worst max abs diff =", worst_max_abs_diff)
        print("Worst mean abs diff =", worst_mean_abs_diff)

        print_shape_comparison_table(torch_shapes, ttnn_shapes)

        if failed_iters:
            print("\n[FAILED ITERS]")
            for item in failed_iters:
                print(item)
            raise AssertionError(
                f"Stress test failed: {len(failed_iters)} / {NUM_ITERS} iterations had PCC <= {PCC_THRESHOLD}"
            )

        print("\n[PASS] Stress PCC test PASSED")



    finally:
        ttnn.close_device(ttnn_device)
        print("[DONE] TT device closed")


if __name__ == "__main__":
    main()



