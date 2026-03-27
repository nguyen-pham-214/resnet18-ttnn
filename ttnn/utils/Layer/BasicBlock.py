from dataclasses import dataclass
from typing import Optional

import ttnn


@dataclass
class BasicBlockWeights:
    conv1_weight: ttnn.Tensor
    conv1_bias: ttnn.Tensor

    conv2_weight: ttnn.Tensor
    conv2_bias: ttnn.Tensor

    shortcut_conv_weight: Optional[ttnn.Tensor] = None
    shortcut_conv_bias: Optional[ttnn.Tensor] = None



class BasicBlock:
    KERNEL_SIZE = (3, 3)
    SHORTCUT_KERNEL_SIZE = (1, 1)

    def __init__(
        self,
        *,
        weights: BasicBlockWeights,
        device,
        in_channels: int,
        out_channels: int,
        batch_size: int,
        input_height: int,
        input_width: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        dtype=None,

        conv1_config=None,
        conv2_config=None,
        shortcut_conv_config=None,

        layer_id=None,

    ) -> None:
        self.weights = weights
        self.device = device

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dtype = dtype

        self.conv1_config = conv1_config
        self.conv2_config = conv2_config
        self.shortcut_conv_config = shortcut_conv_config

        self.conv1_output_height = self._conv_out_dim(
            input_size=self.input_height,
            kernel_size=self.KERNEL_SIZE[0],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        self.conv1_output_width = self._conv_out_dim(
            input_size=self.input_width,
            kernel_size=self.KERNEL_SIZE[1],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        self.output_height = self.conv1_output_height
        self.output_width = self.conv1_output_width

        self.use_projection = self.weights.shortcut_conv_weight is not None

        self.interleaved_l1 = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.L1,
        )

        self.interleaved_dram = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.DRAM,
        )

        self.layer_id = layer_id

    # calc the output dimension after convolution
    @staticmethod
    def _conv_out_dim(
        *,
        input_size: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int = 1,
    ) -> int:
        return ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


    def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        needs_projection = (self.stride != 1) or (self.in_channels != self.out_channels)
        assert self.use_projection == needs_projection, (
            f"use_projection={self.use_projection}, "
            f"stride={self.stride}, in_channels={self.in_channels}, out_channels={self.out_channels}"
        )

        if not self.use_projection:
            identity = input_tensor
        else:
            # input_tensor: HEIGHT_SHARDED + L1
            # self.weights.shortcut_conv_weight: INTERLEAVED + DRAM
            # self.weights.shortcut_conv_bias: INTERLEAVED + DRAM
            shortcut_input = ttnn.to_memory_config(input_tensor, self.interleaved_dram)

            identity = ttnn.conv2d(
                input_tensor=shortcut_input,
                weight_tensor=self.weights.shortcut_conv_weight,
                bias_tensor=self.weights.shortcut_conv_bias,
                device=self.device,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=self.batch_size,
                input_height=self.input_height,
                input_width=self.input_width,
                kernel_size=self.SHORTCUT_KERNEL_SIZE,
                stride=(self.stride, self.stride),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                dtype=self.dtype,
                conv_config=self.shortcut_conv_config,
                return_output_dim=False,
                return_weights_and_bias=False,
            )
            del shortcut_input

        conv1_out = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights.conv1_weight,
            bias_tensor=self.weights.conv1_bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=self.batch_size,
            input_height=self.input_height,
            input_width=self.input_width,
            kernel_size=self.KERNEL_SIZE,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation),
            groups=self.groups,
            dtype=self.dtype,
            conv_config=self.conv1_config,
            return_output_dim=False,
            return_weights_and_bias=False,
        )
    
        if self.layer_id == 4:
            conv2_in = ttnn.to_memory_config(conv1_out, self.interleaved_dram)
            del conv1_out
        else:
            conv2_in = conv1_out        

        out = ttnn.conv2d(
            input_tensor=conv2_in,
            weight_tensor=self.weights.conv2_weight,
            bias_tensor=self.weights.conv2_bias,
            device=self.device,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            batch_size=self.batch_size,
            input_height=self.conv1_output_height,
            input_width=self.conv1_output_width,
            kernel_size=self.KERNEL_SIZE,
            stride=(1, 1),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation),
            groups=self.groups,
            dtype=self.dtype,
            conv_config=self.conv2_config,
            return_output_dim=False,
            return_weights_and_bias=False,
        )

        del conv2_in

        out = ttnn.add(
            out,
            identity,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
        )
        
        return out
