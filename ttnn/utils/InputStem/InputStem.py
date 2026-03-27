from dataclasses import dataclass
import ttnn

@dataclass
class InputStemWeights:
    conv_weight: ttnn.Tensor
    conv_bias: ttnn.Tensor | None


class InputStem:
    IN_CHANNELS = 3
    OUT_CHANNELS = 64
    KERNEL_SIZE = (3, 3)
    STRIDE = (1, 1)
    PADDING = (1, 1)
    DILATION = (1, 1)
    GROUPS = 1

    def __init__(
        self,
        *,
        weights: InputStemWeights,
        device,
        batch_size: int,
        input_height: int,
        input_width: int,
        dtype=ttnn.bfloat16,
        conv2d_config=None,
    ):
        self.weights = weights
        self.device = device
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.dtype = dtype

        # fold RELU into convolution unless overridden from outside
        self.conv2d_config = conv2d_config

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

        self.output_height = self.conv_output_height
        self.output_width = self.conv_output_width

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
        # breakpoint()
        # print(self.conv2d_config)
        return ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights.conv_weight,
            bias_tensor=self.weights.conv_bias,
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
            conv_config=self.conv2d_config,
            return_output_dim=False,
            return_weights_and_bias=False,
        )