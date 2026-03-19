from dataclasses import dataclass
from pathlib import Path

import torch
import ttnn

from utils.InputStem.InputStem import InputStem, InputStemWeights
from utils.Layer.ResNetLayer import ResNetLayer
from utils.Head.Head import ResNetHead, HeadWeights


@dataclass
class ResNet18Weights:
    stem: InputStemWeights
    layer1: dict[str, ttnn.Tensor]
    layer2: dict[str, ttnn.Tensor]
    layer3: dict[str, ttnn.Tensor]
    layer4: dict[str, ttnn.Tensor]
    head: HeadWeights


def _to_row_major(tensor: torch.Tensor, *, device, dtype):
    tt = ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    return ttnn.to_memory_config(tt, ttnn.DRAM_MEMORY_CONFIG)


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

    
def _build_layer_dict(*, state_dict: dict, layer_id: int, device, dtype):
    layer_dict = {}
    prefix = f"layer{layer_id}."

    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue

        is_bn_tensor = (
            "running_mean" in key
            or "running_var" in key
            or (".bn" in key and (key.endswith(".weight") or key.endswith(".bias")))
            or ".shortcut.1.weight" in key
            or ".shortcut.1.bias" in key
            or ".shortcut.1.running_mean" in key
            or ".shortcut.1.running_var" in key
        )

        if is_bn_tensor:
            layer_dict[key] = _to_tile_bn(
                value.to(torch.bfloat16),
                device=device,
                dtype=dtype,
            )
        else:
            layer_dict[key] = _to_row_major_host(
                value.to(torch.bfloat16),
                dtype=dtype,
            )

    return layer_dict


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

        self.stem = InputStem(
            weights=weights.stem,
            device=device,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            dtype=dtype,
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
            conv2d_config=conv2d_config,
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
            conv2d_config=conv2d_config,
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
            conv2d_config=conv2d_config,
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
            conv2d_config=conv2d_config,
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

        acts["input"] = input_tensor
        shapes["input"] = tuple(input_tensor.shape)

        # input stem
        x = self.stem.forward(input_tensor)
        acts["stem"] = x
        shapes["stem"] = tuple(x.shape)

        # residual layer
        x = self.layer1.forward(x)
        acts["layer1"] = x
        shapes["layer1"] = tuple(x.shape)

        x = self.layer2.forward(x)
        acts["layer2"] = x
        shapes["layer2"] = tuple(x.shape)

        x = self.layer3.forward(x)
        acts["layer3"] = x
        shapes["layer3"] = tuple(x.shape)

        x = self.layer4.forward(x)
        acts["layer4"] = x
        shapes["layer4"] = tuple(x.shape)


        # head classification
        x = self.head.forward(x)
        acts["head"] = x
        shapes["head"] = tuple(x.shape)

        return x, acts, shapes


def load_resnet18_from_torch_checkpoint(
    *,
    weights_path: Path,
    device,
    batch_size: int,
    input_height: int,
    input_width: int,
    num_classes: int,
    dtype,
    conv2d_config=None,
    head_memory_config=None,
):
    state_dict = torch.load(weights_path, map_location="cpu")

    stem_weights = InputStemWeights(
        conv_weight=_to_row_major_host(
            state_dict["conv1.weight"].to(torch.bfloat16),
            dtype=dtype,
        ),
        bn_running_mean=_to_tile_bn(
            state_dict["bn1.running_mean"].to(torch.bfloat16),
            device=device,
            dtype=dtype,
        ),
        bn_running_var=_to_tile_bn(
            state_dict["bn1.running_var"].to(torch.bfloat16),
            device=device,
            dtype=dtype,
        ),
        bn_weight=_to_tile_bn(
            state_dict["bn1.weight"].to(torch.bfloat16),
            device=device,
            dtype=dtype,
        ),
        bn_bias=_to_tile_bn(
            state_dict["bn1.bias"].to(torch.bfloat16),
            device=device,
            dtype=dtype,
        ),
    )

    layer1 = _build_layer_dict(
        state_dict=state_dict,
        layer_id=1,
        device=device,
        dtype=dtype,
    )
    layer2 = _build_layer_dict(
        state_dict=state_dict,
        layer_id=2,
        device=device,
        dtype=dtype,
    )
    layer3 = _build_layer_dict(
        state_dict=state_dict,
        layer_id=3,
        device=device,
        dtype=dtype,
    )
    layer4 = _build_layer_dict(
        state_dict=state_dict,
        layer_id=4,
        device=device,
        dtype=dtype,
    )

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
