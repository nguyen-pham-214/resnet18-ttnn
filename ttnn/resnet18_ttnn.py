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


def _to_tile_bn(tensor: torch.Tensor, *, device, dtype):
    tensor = tensor.reshape(1, 1, 1, -1)
    tt = ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    return ttnn.to_memory_config(tt, ttnn.DRAM_MEMORY_CONFIG)

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

        # convolution config
        stem_conv_config = get_module_conv_configs(conv2d_config, module="conv0")
        layer1_conv_config = get_module_conv_configs(conv2d_config, module="conv1")
        layer2_conv_config = get_module_conv_configs(conv2d_config, module="conv2")
        layer3_conv_config = get_module_conv_configs(conv2d_config, module="conv3")
        layer4_conv_config = get_module_conv_configs(conv2d_config, module="conv4")
        # head_conv_config = get_module_conv_configs(conv2d_config, module="head")
         
        
        self.stem = InputStem(
            weights=weights.stem,
            device=device,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            dtype=dtype,
            conv2d_config=stem_conv_config,
        )

        current_height = self.stem.output_height
        current_width = self.stem.output_width
        current_channels = self.stem.OUT_CHANNELS

        current_channels = 64

        self.layer1 = ResNetLayer(
            layer_id=1,
            weights=weights.layer1,
            device=device,
            in_channels=current_channels,
            batch_size=batch_size,
            input_height=current_height,
            input_width=current_width,
            dtype=dtype,
            conv2d_config=layer1_conv_config,
        )

        current_height = self.layer1.output_height
        current_width = self.layer1.output_width
        current_channels = self.layer1.output_channels

        self.layer2 = ResNetLayer(
            layer_id=2,
            weights=weights.layer2,
            device=device,
            in_channels=current_channels,
            batch_size=batch_size,
            input_height=current_height,
            input_width=current_width,
            dtype=dtype,
            conv2d_config=layer2_conv_config,
        )

        current_height = self.layer2.output_height
        current_width = self.layer2.output_width
        current_channels = self.layer2.output_channels

        self.layer3 = ResNetLayer(
            layer_id=3,
            weights=weights.layer3,
            device=device,
            in_channels=current_channels,
            batch_size=batch_size,
            input_height=current_height,
            input_width=current_width,
            dtype=dtype,
            conv2d_config=layer3_conv_config,
        )

        current_height = self.layer3.output_height
        current_width = self.layer3.output_width
        current_channels = self.layer3.output_channels

        self.layer4 = ResNetLayer(
            layer_id=4,
            weights=weights.layer4,
            device=device,
            in_channels=current_channels,
            batch_size=batch_size,
            input_height=current_height,
            input_width=current_width,
            dtype=dtype,
            conv2d_config=layer4_conv_config,
        )

        self.head = ResNetHead(
            weights=weights.head,
            batch_size=batch_size,
            num_classes=num_classes,
            input_height=self.layer4.output_height,
            input_width=self.layer4.output_width,
            memory_config=head_memory_config,
        )

    def forward(self, input_tensor: ttnn.Tensor):


        # to track the shape and activation per layer for debuging
        acts = {}
        shapes = {}

        acts["input"] = ttnn.to_torch(input_tensor).detach().cpu().float()
        shapes["input"] = tuple(input_tensor.shape)
        # print(f"MEMORY CONFIG - INPUT: {ttnn.get_memory_config(input_tensor)}")
        # breakpoint()

        # input stem
        x = self.stem(input_tensor)
        acts["stem"] = ttnn.to_torch(x).detach().cpu().float()
        shapes["stem"] = tuple(x.shape)
        # print(f"MEMORY CONFIG - STEM: {ttnn.get_memory_config(x)}")

        # breakpoint()
        # residual layer
        x = self.layer1(x)
        acts["layer1"] = ttnn.to_torch(x).detach().cpu().float()
        shapes["layer1"] = tuple(x.shape)
        # print(f"MEMORY CONFIG - LAYER 1: {ttnn.get_memory_config(x)}")
        # breakpoint()

        x = self.layer2(x)
        acts["layer2"] = ttnn.to_torch(x).detach().cpu().float()
        shapes["layer2"] = tuple(x.shape)
        # print(f"MEMORY CONFIG - LAYER 2: {ttnn.get_memory_config(x)}")
        # breakpoint()

        x = self.layer3(x)
        acts["layer3"] = ttnn.to_torch(x).detach().cpu().float()
        shapes["layer3"] = tuple(x.shape)
        # print(f"MEMORY CONFIG - LAYER 3: {ttnn.get_memory_config(x)}")
        # breakpoint()

        x = self.layer4(x)
        acts["layer4"] = ttnn.to_torch(x).detach().cpu().float()
        shapes["layer4"] = tuple(x.shape)
        # print(f"MEMORY CONFIG - LAYER 4: {ttnn.get_memory_config(x)}")
        # breakpoint()

        # head classification
        x = self.head(x)
        acts["head"] = ttnn.to_torch(x).detach().cpu().float()
        shapes["head"] = tuple(x.shape)
        # print(f"MEMORY CONFIG - HEAD: {ttnn.get_memory_config(x)}")
        # breakpoint()

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
