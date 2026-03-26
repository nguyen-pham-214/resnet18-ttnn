from dataclasses import dataclass
from typing import Optional

import ttnn


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

def _dump_tensor(tag, x):
    print(f"===== {tag} =====")
    print("shape:", x.shape)
    print("dtype:", x.dtype)
    print("layout:", x.layout)
    print("mem:", x.memory_config())
    try:
        mc = x.memory_config()
        if hasattr(mc, "shard_spec") and mc.shard_spec is not None:
            print("shard grid:", mc.shard_spec.grid)
            print("shard shape:", mc.shard_spec.shape)
    except Exception as e:
        print("shard inspect failed:", e)
    print("=================")


def _dump_conv_meta(self, name, weight, bias, in_ch, out_ch, h, w, stride, kernel, cfg):
    print(f"===== {name} META =====")
    print("layer_id:", self.layer_id)
    print("in_channels:", in_ch)
    print("out_channels:", out_ch)
    print("batch_size:", self.batch_size)
    print("input_height:", h)
    print("input_width:", w)
    print("stride:", stride)
    print("kernel:", kernel)
    print("groups:", self.groups)
    print("dtype:", self.dtype)
    print("config:", cfg)
    print("weight shape:", weight.shape)
    print("weight dtype:", weight.dtype)
    print("weight mem:", weight.memory_config())
    if bias is not None:
        print("bias shape:", bias.shape)
        print("bias dtype:", bias.dtype)
        print("bias mem:", bias.memory_config())
    else:
        print("bias: None")
    print("======================")

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

     
    # def _forward_shortcut(self, input_tensor):
    #     if not self.use_projection:
    #         print("shortcut: identity path")
    #         # return input_tensor
    #         return ttnn.to_memory_config(input_tensor, self.interleaved_l1)

    #     print("shortcut: projection path start")
    #     # breakpoint()
    #     # input_tensor = ttnn.to_memory_config(input_tensor, self.interleaved_l1)
    #     identity = ttnn.conv2d(
    #         input_tensor=input_tensor,
    #         weight_tensor=self.weights.shortcut_conv_weight,
    #         bias_tensor=self.weights.shortcut_conv_bias,
    #         device=self.device,
    #         in_channels=self.in_channels,
    #         out_channels=self.out_channels,
    #         batch_size=self.batch_size,
    #         input_height=self.input_height,
    #         input_width=self.input_width,
    #         kernel_size=self.SHORTCUT_KERNEL_SIZE,
    #         stride=(self.stride, self.stride),
    #         padding=(0, 0),
    #         dilation=(1, 1),
    #         groups=1,
    #         dtype=self.dtype,
    #         conv_config=self.shortcut_conv_config,
    #         return_output_dim=False,
    #         return_weights_and_bias=False,
    #     )
    #     print("shortcut: projection path done")
    #     print("shortcut mem:", identity.memory_config())
    #     print("shortcut shape:", identity.shape)
    #     return identity

    # def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    #     breakpoint()
    #     identity = self._forward_shortcut(input_tensor)
    #     print("shortcut done")

    #     # breakpoint()
    #     out = ttnn.conv2d(
    #         input_tensor=input_tensor,
    #         weight_tensor=self.weights.conv1_weight,
    #         bias_tensor=self.weights.conv1_bias,
    #         device=self.device,
    #         in_channels=self.in_channels,
    #         out_channels=self.out_channels,
    #         batch_size=self.batch_size,
    #         input_height=self.input_height,
    #         input_width=self.input_width,
    #         kernel_size=self.KERNEL_SIZE,
    #         stride=(self.stride, self.stride),
    #         padding=(self.padding, self.padding),
    #         dilation=(self.dilation, self.dilation),
    #         groups=self.groups,
    #         dtype=self.dtype,
    #         conv_config=self.conv1_config,
    #         return_output_dim=False,
    #         return_weights_and_bias=False,

    #         # memory_config=self.interleaved_l1,

    #         # slice_config=ttnn.Conv2dL1FullSliceConfig,
    #     )
    #     # ttnn.synchronize_device(self.device)
    #     print("conv1 done")

    #     # breakpoint()
    #     out = ttnn.conv2d(
    #         input_tensor=out,
    #         weight_tensor=self.weights.conv2_weight,
    #         bias_tensor=self.weights.conv2_bias,
    #         device=self.device,
    #         in_channels=self.out_channels,
    #         out_channels=self.out_channels,
    #         batch_size=self.batch_size,
    #         input_height=self.conv1_output_height,
    #         input_width=self.conv1_output_width,
    #         kernel_size=self.KERNEL_SIZE,
    #         stride=(1, 1),
    #         padding=(self.padding, self.padding),
    #         dilation=(self.dilation, self.dilation),
    #         groups=self.groups,
    #         dtype=self.dtype,
    #         conv_config=self.conv2_config,
    #         return_output_dim=False,
    #         return_weights_and_bias=False,

    #         # memory_config=self.interleaved_l1,

    #         # slice_config=ttnn.Conv2dL1FullSliceConfig,
    #     )
    #     # ttnn.synchronize_device(self.device)
    #     print("conv2 done")


    #     # # Check allocation before add  
    #     # print(f"out allocated: {out.is_allocated()}")  
    #     # print(f"identity allocated: {identity.is_allocated()}")  
    #     # print(f"out memory config: {out.memory_config()}")  
    #     # print(f"identity memory config: {identity.memory_config()}")  

    #     # breakpoint()
    #     out = ttnn.add(
    #         out,
    #         identity,
    #         activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
    #     )
    #     # ttnn.synchronize_device(self.device)
    #     print("add done")

    #     return out


    def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        needs_projection = (self.stride != 1) or (self.in_channels != self.out_channels)
        assert self.use_projection == needs_projection, (
            f"use_projection={self.use_projection}, "
            f"stride={self.stride}, in_channels={self.in_channels}, out_channels={self.out_channels}"
        )

        # print(f"\n========== ENTER BLOCK layer={self.layer_id} ==========")
        # _dump_tensor("block input", input_tensor)

        if not self.use_projection:
            # print("shortcut: identity path")
            identity = input_tensor
        else:
            shortcut_input = ttnn.to_memory_config(input_tensor, self.interleaved_dram)
            # _dump_tensor("shortcut_input", shortcut_input)
            # _dump_conv_meta(
            #     self,
            #     "shortcut",
            #     self.weights.shortcut_conv_weight,
            #     self.weights.shortcut_conv_bias,
            #     self.in_channels,
            #     self.out_channels,
            #     self.input_height,
            #     self.input_width,
            #     (self.stride, self.stride),
            #     self.SHORTCUT_KERNEL_SIZE,
            #     self.shortcut_conv_config,
            # )

            # print("shortcut: projection path start")
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
            # print("shortcut: projection launched")
            # ttnn.synchronize_device(self.device)
            # print("shortcut: projection synced")
            # _dump_tensor("identity after shortcut", identity)
            del shortcut_input

        # _dump_conv_meta(
        #     self,
        #     "conv1",
        #     self.weights.conv1_weight,
        #     self.weights.conv1_bias,
        #     self.in_channels,
        #     self.out_channels,
        #     self.input_height,
        #     self.input_width,
        #     (self.stride, self.stride),
        #     self.KERNEL_SIZE,
        #     self.conv1_config,
        # )

        # print("before conv1")
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
        # print("conv1 launched")
        # ttnn.synchronize_device(self.device)
        # print("conv1 synced")
        # _dump_tensor("conv1_out", conv1_out)

        # # Probe A: thử input hiện tại của conv2
        # conv2_in = conv1_out
        if self.layer_id == 4:
            conv2_in = ttnn.to_memory_config(conv1_out, self.interleaved_dram)
            # _dump_tensor("conv2_in dram", conv2_in)
            del conv1_out
        else:
            conv2_in = conv1_out        

        # _dump_conv_meta(
        #     self,
        #     "conv2",
        #     self.weights.conv2_weight,
        #     self.weights.conv2_bias,
        #     self.out_channels,
        #     self.out_channels,
        #     self.conv1_output_height,
        #     self.conv1_output_width,
        #     (1, 1),
        #     self.KERNEL_SIZE,
        #     self.conv2_config,
        # )
        # _dump_tensor("conv2_in original", conv2_in)

        # print("before conv2 (original input)")
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
        # print("conv2 launched")
        # ttnn.synchronize_device(self.device)
        # print("conv2 synced")
        # _dump_tensor("conv2_out", out)

        # del conv1_out
        del conv2_in

        # print("before add")
        # _dump_tensor("out before add", out)
        # _dump_tensor("identity before add", identity)

        out = ttnn.add(
            out,
            identity,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
        )
        # print("add launched")
        # ttnn.synchronize_device(self.device)
        # print("add synced")
        # _dump_tensor("block output", out)

        return out
