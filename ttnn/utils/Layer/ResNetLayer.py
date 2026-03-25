from dataclasses import dataclass

import ttnn

from .BasicBlock import BasicBlock, BasicBlockWeights


@dataclass
class LayerSpec:
    out_channels: int
    num_blocks: int
    first_stride: int


class ResNetLayer:
    LAYER_SPECS = {
        1: LayerSpec(out_channels=64, num_blocks=2, first_stride=1),
        2: LayerSpec(out_channels=128, num_blocks=2, first_stride=2),
        3: LayerSpec(out_channels=256, num_blocks=2, first_stride=2),
        4: LayerSpec(out_channels=512, num_blocks=2, first_stride=2),
    }

    def __init__(
        self,
        *,
        layer_id: int,
        weights: dict[str, ttnn.Tensor],
        device,
        in_channels: int,
        batch_size: int,
        input_height: int,
        input_width: int,
        dtype=None,
        conv2d_config=None,
    ) -> None:
        if layer_id not in self.LAYER_SPECS:
            raise ValueError(f"Unsupported layer_id={layer_id}, expected one of {list(self.LAYER_SPECS.keys())}")

        self.layer_id = layer_id
        self.weights = weights
        self.device = device
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.dtype = dtype
        self.conv2d_config = conv2d_config or {}

        spec = self.LAYER_SPECS[layer_id]
        self.out_channels = spec.out_channels
        self.num_blocks = spec.num_blocks
        self.first_stride = spec.first_stride

        self.blocks: list[BasicBlock] = []
        self.output_height = input_height
        self.output_width = input_width
        self.output_channels = self.out_channels

        self._build()

    def _make_block_weights(self, block_id: int, use_projection: bool) -> BasicBlockWeights:
        prefix = f"layer{self.layer_id}.{block_id}"

        return BasicBlockWeights(
            conv1_weight=self.weights[f"{prefix}.conv1.weight"],
            conv1_bias=self.weights[f"{prefix}.conv1.bias"],
            conv2_weight=self.weights[f"{prefix}.conv2.weight"],
            conv2_bias=self.weights[f"{prefix}.conv2.bias"],
            shortcut_conv_weight=self.weights.get(f"{prefix}.shortcut.0.weight") if use_projection else None,
            shortcut_conv_bias=self.weights.get(f"{prefix}.shortcut.0.bias") if use_projection else None,
        )

    def _build(self) -> None:
        current_in_channels = self.in_channels
        current_height = self.input_height
        current_width = self.input_width

        for block_id in range(self.num_blocks):
            stride = self.first_stride if block_id == 0 else 1
            use_projection = (stride != 1) or (current_in_channels != self.out_channels)

            block_weights = self._make_block_weights(
                block_id=block_id,
                use_projection=use_projection,
            )

            config_prefix = f"conv{self.layer_id}.{block_id}"

            block = BasicBlock(
                weights=block_weights,
                device=self.device,
                in_channels=current_in_channels,
                out_channels=self.out_channels,
                batch_size=self.batch_size,
                input_height=current_height,
                input_width=current_width,
                stride=stride,
                padding=1,
                dilation=1,
                groups=1,
                dtype=self.dtype,
                conv1_config=self.conv2d_config.get(f"{config_prefix}.0"),
                conv2_config=self.conv2d_config.get(f"{config_prefix}.1"),
                shortcut_conv_config=self.conv2d_config.get(f"{config_prefix}.shortcut"),
            )

            self.blocks.append(block)

            current_in_channels = self.out_channels
            current_height = block.output_height
            current_width = block.output_width

        self.output_height = current_height
        self.output_width = current_width

    def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        x = input_tensor
        for block in self.blocks:
            x = block(x)
        return x