from dataclasses import dataclass
import ttnn


@dataclass
class InputStemWeights:
    conv_weight: ttnn.Tensor
    conv_bias: ttnn.Tensor | None


class InputStem:
    IN_CHANNELS = 3
    OUT_CHANNELS = 64
    KERNEL_SIZE = (7, 7)
    STRIDE = (2, 2)
    PADDING = (3, 3)
    DILATION = (1, 1)
    GROUPS = 1

    POOL_KERNEL = (3, 3)
    POOL_STRIDE = (2, 2)
    POOL_PADDING = (1, 1)
    POOL_DILATION = (1, 1)

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
        self.conv2d_config = conv2d_config

        conv_out_h = ((input_height + 2 * self.PADDING[0] - self.DILATION[0] * (self.KERNEL_SIZE[0] - 1) - 1) // self.STRIDE[0]) + 1
        conv_out_w = ((input_width + 2 * self.PADDING[1] - self.DILATION[1] * (self.KERNEL_SIZE[1] - 1) - 1) // self.STRIDE[1]) + 1

        self.output_height = ((conv_out_h + 2 * self.POOL_PADDING[0] - self.POOL_DILATION[0] * (self.POOL_KERNEL[0] - 1) - 1) // self.POOL_STRIDE[0]) + 1
        self.output_width = ((conv_out_w + 2 * self.POOL_PADDING[1] - self.POOL_DILATION[1] * (self.POOL_KERNEL[1] - 1) - 1) // self.POOL_STRIDE[1]) + 1

    @staticmethod
    def _out_dim(input_size: int, kernel_size: int, stride: int, padding: int, dilation: int = 1) -> int:
        return ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

    def __call__(self, input_tensor: ttnn.Tensor) -> tuple[ttnn.Tensor, int, int, int]:

        x, (conv_out_h, conv_out_w) = ttnn.conv2d(
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
            return_output_dim=True,
            return_weights_and_bias=False,
        )

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=conv_out_h,
            input_w=conv_out_w,
            channels=self.OUT_CHANNELS,
            kernel_size=list(self.POOL_KERNEL),
            stride=list(self.POOL_STRIDE),
            padding=list(self.POOL_PADDING),
            dilation=list(self.POOL_DILATION),

            output_layout=ttnn.TILE_LAYOUT, 
            reallocate_halo_output=True,
            deallocate_input=True,    
        )

        return x


        