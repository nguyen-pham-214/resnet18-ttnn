from dataclasses import dataclass
from pathlib import Path

import torch
import ttnn

from utils.InputStem.InputStem import InputStem, InputStemWeights
from utils.Layer.ResNetLayer import ResNetLayer
from utils.Head.Head import ResNetHead, HeadWeights
from configs import conv2d_config


@dataclass
class ResNet18Weights:
    stem: InputStemWeights
    layer1: dict[str, ttnn.Tensor]
    layer2: dict[str, ttnn.Tensor]
    layer3: dict[str, ttnn.Tensor]
    layer4: dict[str, ttnn.Tensor]
    head: HeadWeights


def _to_row_major_host(tensor: torch.Tensor, *, dtype):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

def _build_fused_layer_dict(*, state_dict: dict, layer_id: int, dtype):
    layer_dict = {}
    prefix = f"layer{layer_id}."

    # Find block ids present in this layer
    block_ids = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict.keys()
            if key.startswith(prefix)
        }
    )

    for block_id in block_ids:
        block_prefix = f"{prefix}{block_id}"

        # conv1 + bn1
        conv1_w = state_dict[f"{block_prefix}.conv1.weight"].to(torch.bfloat16)
        bn1_mean = state_dict[f"{block_prefix}.bn1.running_mean"].to(torch.bfloat16)
        bn1_var = state_dict[f"{block_prefix}.bn1.running_var"].to(torch.bfloat16)
        bn1_weight = state_dict[f"{block_prefix}.bn1.weight"].to(torch.bfloat16)
        bn1_bias = state_dict[f"{block_prefix}.bn1.bias"].to(torch.bfloat16)

        fused_conv1_w, fused_conv1_b = fold_bn_into_conv(
            conv1_w,
            bn1_mean,
            bn1_var,
            bn1_weight,
            bn1_bias,
            eps=1e-5,
        )

        layer_dict[f"{block_prefix}.conv1.weight"] = _to_row_major_host(
            fused_conv1_w,
            dtype=dtype,
        )
        layer_dict[f"{block_prefix}.conv1.bias"] = _to_row_major_host(
            fused_conv1_b.reshape(1, 1, 1, -1),
            dtype=dtype,
        )

        # conv2 + bn2
        conv2_w = state_dict[f"{block_prefix}.conv2.weight"].to(torch.bfloat16)
        bn2_mean = state_dict[f"{block_prefix}.bn2.running_mean"].to(torch.bfloat16)
        bn2_var = state_dict[f"{block_prefix}.bn2.running_var"].to(torch.bfloat16)
        bn2_weight = state_dict[f"{block_prefix}.bn2.weight"].to(torch.bfloat16)
        bn2_bias = state_dict[f"{block_prefix}.bn2.bias"].to(torch.bfloat16)

        fused_conv2_w, fused_conv2_b = fold_bn_into_conv(
            conv2_w,
            bn2_mean,
            bn2_var,
            bn2_weight,
            bn2_bias,
            eps=1e-5,
        )

        layer_dict[f"{block_prefix}.conv2.weight"] = _to_row_major_host(
            fused_conv2_w,
            dtype=dtype,
        )
        layer_dict[f"{block_prefix}.conv2.bias"] = _to_row_major_host(
            fused_conv2_b.reshape(1, 1, 1, -1),
            dtype=dtype,
        )

        # optional shortcut.0 + shortcut.1
        shortcut_conv_key = f"{block_prefix}.shortcut.0.weight"
        if shortcut_conv_key in state_dict:
            shortcut_w = state_dict[shortcut_conv_key].to(torch.bfloat16)
            shortcut_mean = state_dict[f"{block_prefix}.shortcut.1.running_mean"].to(torch.bfloat16)
            shortcut_var = state_dict[f"{block_prefix}.shortcut.1.running_var"].to(torch.bfloat16)
            shortcut_weight = state_dict[f"{block_prefix}.shortcut.1.weight"].to(torch.bfloat16)
            shortcut_bias = state_dict[f"{block_prefix}.shortcut.1.bias"].to(torch.bfloat16)

            fused_shortcut_w, fused_shortcut_b = fold_bn_into_conv(
                shortcut_w,
                shortcut_mean,
                shortcut_var,
                shortcut_weight,
                shortcut_bias,
                eps=1e-5,
            )

            layer_dict[f"{block_prefix}.shortcut.0.weight"] = _to_row_major_host(
                fused_shortcut_w,
                dtype=dtype,
            )
            layer_dict[f"{block_prefix}.shortcut.0.bias"] = _to_row_major_host(
                fused_shortcut_b.reshape(1, 1, 1, -1),
                dtype=dtype,
            )

    return layer_dict

def get_module_conv_configs(
    conv2d_config: dict | None,
    *,
    module: str,
    normalize_keys: bool = True,
):
    """
    Extract conv configs for a given module.

    Args:
        conv2d_config: full config dict
        module: e.g. "conv0", "conv1", "conv2", "head"
        normalize_keys:
            - True  -> keep keys with module prefix (conv1.0.0)
            - False -> keep full original keys (same behavior)

    Returns:
        dict for modules with sub-structure (layers/head)
        single config or None for flat modules (e.g. conv0)
    """
    if conv2d_config is None:
        return None if module == "conv0" else {}

    # Stem (single entry)
    if module == "conv0":
        return conv2d_config.get("conv0", None)

    prefix = f"{module}."

    out = {}
    for key, value in conv2d_config.items():
        if key.startswith(prefix):
            out[key] = value

    return out

def print_mem(tag, device):  
    print(f"\n===== MEMORY @ {tag} =====")  
    try:  
        # This generates CSV files, not console output  
        ttnn.device.dump_device_memory_state(device, prefix=f"layer_{tag}_")  
        print(f"Memory report generated for {tag}")  
    except Exception as e:  
        print(f"Error: {e}")  
          
    # For immediate console output, try:  
    try:  
        memory_view = ttnn.device.get_memory_view(device, ttnn.BufferType.L1)  
        print(f"L1 Memory - Total: {memory_view.total_bytes_per_bank}, "  
              f"Allocated: {memory_view.total_bytes_allocated_per_bank}, "  
              f"Free: {memory_view.total_bytes_free_per_bank}")  
    except Exception as e:  
        print(f"Could not get memory view: {e}")


def test_conv_layout(device):
    import ttnn
    import torch

    # ----------------------------
    # Create random input + weight
    # ----------------------------
    N, H, W, C = 1, 32, 32, 3
    OC = 8

    torch_input = torch.randn((N, H, W, C), dtype=torch.float32)
    torch_weight = torch.randn((OC, C, 3, 3), dtype=torch.float32)

    input_tensor = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16)
    weight_tensor = ttnn.from_torch(torch_weight, device=device, dtype=ttnn.bfloat16)

    def run_conv(x, label):
        try:
            ttnn.conv2d(
                input_tensor=x,
                weight_tensor=weight_tensor,
                device=device,
                in_channels=C,
                out_channels=OC,
                batch_size=N,
                input_height=H,
                input_width=W,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=1,
                dtype=ttnn.bfloat16,
                return_output_dim=False,
            )
            print(f"[PASS] {label}")
        except Exception as e:
            print(f"[FAIL] {label}")
            print("   ", e)

    # ----------------------------
    # INTERLEAVED
    # ----------------------------
    interleaved_mem = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    x_interleaved = ttnn.to_memory_config(input_tensor, interleaved_mem)

    run_conv(x_interleaved, "INTERLEAVED")

    # ----------------------------
    # HEIGHT SHARDED
    # ----------------------------
    sharded_mem = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )
    x_sharded = ttnn.to_memory_config(input_tensor, sharded_mem)

    run_conv(x_sharded, "HEIGHT_SHARDED")

class ResNet18:
    def __init__(
        self,
        *,
        weights: ResNet18Weights,
        device,
        batch_size: int,
        input_height: int,
        input_width: int,
        num_classes: int,
        dtype=None,
        conv2d_config=None,
        head_memory_config=None,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.dtype = dtype

        stem_conv_config = get_module_conv_configs(conv2d_config, module="conv0")
        layer1_conv_config = get_module_conv_configs(conv2d_config, module="conv1")
        layer2_conv_config = get_module_conv_configs(conv2d_config, module="conv2")
        layer3_conv_config = get_module_conv_configs(conv2d_config, module="conv3")
        layer4_conv_config = get_module_conv_configs(conv2d_config, module="conv4")

        self.stem = InputStem(
            weights=weights.stem,
            device=device,
            batch_size=self.batch_size,
            input_height=input_height,
            input_width=input_width,
            dtype=dtype,
            conv2d_config=stem_conv_config,
        )

        # Build layer1 from stem's known output spec for ImageNet stem:
        # conv7x7 s2 -> maxpool3x3 s2, so stem returns 64 x 56 x 56 for 224x224 input.
        current_channels = self.stem.OUT_CHANNELS
        current_height = self.stem.output_height
        current_width = self.stem.output_width

        self.layer1 = ResNetLayer(
            layer_id=1,
            weights=weights.layer1,
            device=device,
            in_channels=current_channels,
            batch_size=self.batch_size,
            input_height=current_height,
            input_width=current_width,
            dtype=dtype,
            conv2d_config=layer1_conv_config,
        )
        current_channels = self.layer1.output_channels
        current_height = self.layer1.output_height
        current_width = self.layer1.output_width

        self.layer2 = ResNetLayer(
            layer_id=2,
            weights=weights.layer2,
            device=device,
            in_channels=current_channels,
            batch_size=self.batch_size,
            input_height=current_height,
            input_width=current_width,
            dtype=dtype,
            conv2d_config=layer2_conv_config,
        )
        current_channels = self.layer2.output_channels
        current_height = self.layer2.output_height
        current_width = self.layer2.output_width
        # current_height = 28
        # current_width = 28

        # print("=====After layer2 build: current_channels =", current_channels, "current_height =", current_height, "current_width =", current_width)

        self.layer3 = ResNetLayer(
            layer_id=3,
            weights=weights.layer3,
            device=device,
            in_channels=current_channels,
            batch_size=self.batch_size,
            input_height=current_height,
            input_width=current_width,
            dtype=dtype,
            conv2d_config=layer3_conv_config,
        )
        current_channels = self.layer3.output_channels
        current_height = self.layer3.output_height
        current_width = self.layer3.output_width
        # current_height = 14
        # current_width = 14

        self.layer4 = ResNetLayer(
            layer_id=4,
            weights=weights.layer4,
            device=device,
            in_channels=current_channels,
            batch_size=self.batch_size,
            input_height=current_height,
            input_width=current_width,
            dtype=dtype,
            conv2d_config=layer4_conv_config,
        )
        current_channels = self.layer4.output_channels
        current_height = self.layer4.output_height
        current_width = self.layer4.output_width
        # current_height = 7
        # current_width = 7

        self.head = ResNetHead(
            weights=weights.head,
            batch_size=self.batch_size,
            in_features=current_channels,
            input_height=current_height,
            input_width=current_width,
            num_classes=num_classes,
            dtype=dtype if dtype is not None else ttnn.bfloat16,
            memory_config=head_memory_config,
        )


    def forward(self, input_tensor: ttnn.Tensor):
        device = input_tensor.device()
        # test_conv_layout(device)
        

        acts = {}
        shapes = {}

        print(f"Input shape: (N, C, H, W) = ({self.batch_size}, {input_tensor.shape[1]}, {input_tensor.shape[2]}, {input_tensor.shape[3]})")
        
        print("\nStarting stem")
        # breakpoint()
        # x, c, h, w = self.stem(input_tensor)
        x = self.stem(input_tensor)
        # print(f"After stem: shape = ({self.batch_size}, {c}, {h}, {w})")
        acts["stem"] = x
        # shapes["stem"] = (c, h, w)

        print("\n\nStarting layer1")
        # breakpoint()
        x, c, h, w = self.layer1(x)
        print(f"After layer1: shape = ({self.batch_size}, {c}, {h}, {w})")
        acts["layer1"] = x
        shapes["layer1"] = (c, h, w)
        # print_mem("after layer1", device)

        print("\n\nStarting layer2")
        # breakpoint()
        x, c, h, w = self.layer2(x)
        print(f"After layer2: shape = ({self.batch_size}, {c}, {h}, {w})")
        acts["layer2"] = x
        shapes["layer2"] = (c, h, w)
        # print_mem("after layer2", device)

        print("\n\nStarting layer3")
        # breakpoint()
        x, c, h, w = self.layer3(x)
        print(f"After layer3: shape = ({self.batch_size}, {c}, {h}, {w})")
        acts["layer3"] = x
        shapes["layer3"] = (c, h, w)
        # print_mem("after layer3", device)
        # print(f"===== Before layer 4 =====: {x.shape}")

        print("\n\nStarting layer4")
        # breakpoint()
        x, c, h, w = self.layer4(x)
        print(f"After layer4: shape = ({self.batch_size}, {c}, {h}, {w})")
        acts["layer4"] = x
        shapes["layer4"] = (c, h, w)
        # print_mem("after layer4", device)
        
        # print(f"===== Before head =====: {x.shape}")

        print("\n\nStarting head")
        # breakpoint()
        x = self.head(x)
        # c, h, w = self.head.final_dimension
        # print(f"After head: shape = ({self.batch_size}, {c}, {h}, {w})")
        # acts["avgpool"] = self.head.debug_avgpool
        # acts["flatten"] = self.head.debug_flatten
        acts["head"] = x
        # shapes["head"] = (c, h, w)

        return x, acts, shapes


def fold_bn_into_conv(
    conv_w: torch.Tensor,
    bn_mean: torch.Tensor,
    bn_var: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    eps: float,
    conv_bias: torch.Tensor | None = None,
):
    # conv_w: [out_channels, in_channels, kH, kW]
    # conv_bias: [out_channels] or None
    if conv_bias is None:
        conv_bias = torch.zeros(
            conv_w.shape[0],
            dtype=conv_w.dtype,
            device=conv_w.device,
        )

    scale = bn_weight / torch.sqrt(bn_var + eps)           # [C_out]
    fused_w = conv_w * scale[:, None, None, None]
    fused_b = bn_bias + (conv_bias - bn_mean) * scale

    return fused_w, fused_b

def load_resnet18_from_torch_checkpoint(
    *,
    weights_path: Path,
    device,
    batch_size: int,
    input_height: int,
    input_width: int,
    num_classes: int,
    dtype,
    conv2d_config=conv2d_config,
    head_memory_config=None,
):
    state_dict = torch.load(weights_path, map_location="cpu")

    conv1_weight = state_dict["conv1.weight"].to(torch.bfloat16)
    bn1_running_mean = state_dict["bn1.running_mean"].to(torch.bfloat16)
    bn1_running_var = state_dict["bn1.running_var"].to(torch.bfloat16)
    bn1_weight = state_dict["bn1.weight"].to(torch.bfloat16)
    bn1_bias = state_dict["bn1.bias"].to(torch.bfloat16)

    fused_conv1_weight, fused_conv1_bias = fold_bn_into_conv(
        conv1_weight,
        bn1_running_mean,
        bn1_running_var,
        bn1_weight,
        bn1_bias,
        eps=1e-5,
    )

    stem_weights = InputStemWeights(
        conv_weight=_to_row_major_host(
            fused_conv1_weight,
            dtype=dtype,
        ),
        conv_bias=_to_row_major_host(
            fused_conv1_bias.reshape(1, 1, 1, -1),
            dtype=dtype,
        ),
    )

    layer1 = _build_fused_layer_dict(state_dict=state_dict, layer_id=1, dtype=dtype)
    layer2 = _build_fused_layer_dict(state_dict=state_dict, layer_id=2, dtype=dtype)
    layer3 = _build_fused_layer_dict(state_dict=state_dict, layer_id=3, dtype=dtype)
    layer4 = _build_fused_layer_dict(state_dict=state_dict, layer_id=4, dtype=dtype)

    fc_weight_key = "fc.weight" if "fc.weight" in state_dict else "linear.weight"
    fc_bias_key = "fc.bias" if "fc.bias" in state_dict else "linear.bias"

    head_weights = HeadWeights(
        fc_weight=ttnn.from_torch(
            state_dict[fc_weight_key].to(torch.bfloat16).transpose(0, 1),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        ),
        fc_bias=ttnn.from_torch(
            state_dict[fc_bias_key].to(torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        ) if fc_bias_key in state_dict else None
    )

    weights = ResNet18Weights(
        stem=stem_weights,
        layer1=layer1,
        layer2=layer2,
        layer3=layer3,
        layer4=layer4,
        head=head_weights,
    )

    return ResNet18(
        weights=weights,
        device=device,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        num_classes=num_classes,
        dtype=dtype,
        conv2d_config=conv2d_config,
        head_memory_config=head_memory_config,
    )
