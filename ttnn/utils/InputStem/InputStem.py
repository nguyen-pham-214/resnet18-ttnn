from dataclasses import dataclass

import ttnn


@dataclass
class InputStemWeights:
    conv_weight: ttnn.Tensor
    bn_running_mean: ttnn.Tensor
    bn_running_var: ttnn.Tensor
    bn_weight: ttnn.Tensor
    bn_bias: ttnn.Tensor


class InputStem:
    IN_CHANNELS = 3
    OUT_CHANNELS = 64
    # KERNEL_SIZE = (7, 7)
    # STRIDE = (2, 2)
    # PADDING = (3, 3)
    KERNEL_SIZE = (3, 3)
    STRIDE = (1, 1)
    PADDING = (1, 1)
    DILATION = (1, 1)
    GROUPS = 1

    # # MaxPool after ReLU
    POOL_KERNEL_SIZE = (3, 3)
    POOL_STRIDE = (2, 2)
    POOL_PADDING = (1, 1)
    POOL_DILATION = (1, 1)

    BN_EPS = 1e-5
    BN_MOMENTUM = 0.1

    def __init__(
        self,
        *,
        weights: InputStemWeights,
        device,
        batch_size: int,
        input_height: int,
        input_width: int,
        dtype=ttnn.bfloat16,
        conv_config=None,
    ) -> None:
        self.weights = weights
        self.device = device
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.dtype = dtype
        self.conv_config = conv_config

        self.conv_output_height = self._conv_out_dim(
            input_size=input_height,
            kernel_size=self.KERNEL_SIZE[0],
            stride=self.STRIDE[0],
            padding=self.PADDING[0],
            dilation=self.DILATION[0],
        )
        self.conv_output_width = self._conv_out_dim(
            input_size=input_width,
            kernel_size=self.KERNEL_SIZE[1],
            stride=self.STRIDE[1],
            padding=self.PADDING[1],
            dilation=self.DILATION[1],
        )

        # self.output_height = self._conv_out_dim(
        #     input_size=self.conv_output_height,
        #     kernel_size=self.POOL_KERNEL_SIZE[0],
        #     stride=self.POOL_STRIDE[0],
        #     padding=self.POOL_PADDING[0],
        #     dilation=self.POOL_DILATION[0],
        # )
        
        # self.output_width = self._conv_out_dim(
        #     input_size=self.conv_output_width,
        #     kernel_size=self.POOL_KERNEL_SIZE[1],
        #     stride=self.POOL_STRIDE[1],
        #     padding=self.POOL_PADDING[1],
        #     dilation=self.POOL_DILATION[1],
        # )
        self.output_height = self.conv_output_height
        self.output_width = self.conv_output_width

    # computes the output size of a 1D convolution layer
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

    def _debug_tensor(self, name: str, x):
        print(f"\n[{name}]")

        # shape
        try:
            print("shape:", x.shape)
        except Exception as e:
            print("shape: <unavailable>", e)

        # memory config
        try:
            print("memory_config:", x.memory_config())
        except Exception as e:
            print("memory_config: <unavailable>", e)

        # layout
        try:
            print("layout:", x.layout)
        except Exception as e:
            print("layout: <unavailable>", e)

        # device
        try:
            print("device:", x.device())
        except Exception:
            try:
                print("device:", x.device)
            except Exception as e:
                print("device: <unavailable>", e)

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        # self._debug_tensor("input", input_tensor)
        x = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights.conv_weight,
            device=self.device,

            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            batch_size=self.batch_size,
            input_height=self.input_height,
            input_width=self.input_width,

            kernel_size=self.KERNEL_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING,
            dilation=self.DILATION,
            groups=self.GROUPS,
            dtype=self.dtype,
            bias_tensor=None,
            # conv_config=self.conv_config,
  
            return_output_dim=False,
            return_weights_and_bias=False,
        )
        # self._debug_tensor("after conv2d", x)
        # print("conv out memory_config:", x.memory_config())
        # print("conv out layout:", x.layout)

        x = ttnn.batch_norm(
            x,
            eps=self.BN_EPS,
            momentum=self.BN_MOMENTUM,
            running_mean=self.weights.bn_running_mean,
            running_var=self.weights.bn_running_var,
            weight=self.weights.bn_weight,
            bias=self.weights.bn_bias,
            training=False,
        )
        # self._debug_tensor("after batch_norm", x)

        x = ttnn.relu(
            input_tensor=x,
            # memory_config=self.memory_config
        )   
        # self._debug_tensor("after relu", x)

        # x = ttnn.max_pool2d(
        #     x,
        #     self.batch_size,
        #     self.conv_output_height,
        #     self.conv_output_width,
        #     self.OUT_CHANNELS,
        #     self.POOL_KERNEL_SIZE,
        #     self.POOL_STRIDE,
        #     self.POOL_PADDING,
        #     self.POOL_DILATION,
        # )
        # self._debug_tensor("after max_pool2d", x)

        return x