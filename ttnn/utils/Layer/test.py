import torch
import ttnn

from BasicBlock import BasicBlock, BasicBlockWeights
from ResNetLayer import ResNetLayer


def make_bn_tensor(shape, *, dtype=torch.bfloat16, positive=False):
    if positive:
        return torch.rand(shape, dtype=dtype) + 1.0
    return torch.randn(shape, dtype=dtype)


def to_device_row_major(torch_tensor, *, device, dtype):
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


def to_device_tile(torch_tensor, *, device, dtype):
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def create_basicblock_weights(
    *,
    device,
    dtype,
    in_channels,
    out_channels,
    use_projection,
):
    # Conv weights: [out_channels, in_channels, kH, kW]
    torch_conv1_weight = torch.randn(
        (out_channels, in_channels, 3, 3),
        dtype=torch.bfloat16,
    )
    torch_conv2_weight = torch.randn(
        (out_channels, out_channels, 3, 3),
        dtype=torch.bfloat16,
    )

    # BN params must be rank-4 for ttnn.batch_norm: [1, 1, 1, C]
    torch_bn1_running_mean = make_bn_tensor((1, 1, 1, out_channels))
    torch_bn1_running_var = make_bn_tensor((1, 1, 1, out_channels), positive=True)
    torch_bn1_weight = make_bn_tensor((1, 1, 1, out_channels))
    torch_bn1_bias = make_bn_tensor((1, 1, 1, out_channels))

    torch_bn2_running_mean = make_bn_tensor((1, 1, 1, out_channels))
    torch_bn2_running_var = make_bn_tensor((1, 1, 1, out_channels), positive=True)
    torch_bn2_weight = make_bn_tensor((1, 1, 1, out_channels))
    torch_bn2_bias = make_bn_tensor((1, 1, 1, out_channels))

    weights = BasicBlockWeights(
        conv1_weight=to_device_row_major(torch_conv1_weight, device=device, dtype=dtype),
        conv2_weight=to_device_row_major(torch_conv2_weight, device=device, dtype=dtype),
        bn1_running_mean=to_device_tile(torch_bn1_running_mean, device=device, dtype=dtype),
        bn1_running_var=to_device_tile(torch_bn1_running_var, device=device, dtype=dtype),
        bn1_weight=to_device_tile(torch_bn1_weight, device=device, dtype=dtype),
        bn1_bias=to_device_tile(torch_bn1_bias, device=device, dtype=dtype),
        bn2_running_mean=to_device_tile(torch_bn2_running_mean, device=device, dtype=dtype),
        bn2_running_var=to_device_tile(torch_bn2_running_var, device=device, dtype=dtype),
        bn2_weight=to_device_tile(torch_bn2_weight, device=device, dtype=dtype),
        bn2_bias=to_device_tile(torch_bn2_bias, device=device, dtype=dtype),
        bias1_tensor=None,
        bias2_tensor=None,
    )

    if use_projection:
        torch_shortcut_conv_weight = torch.randn(
            (out_channels, in_channels, 1, 1),
            dtype=torch.bfloat16,
        )

        torch_shortcut_bn_running_mean = make_bn_tensor((1, 1, 1, out_channels))
        torch_shortcut_bn_running_var = make_bn_tensor((1, 1, 1, out_channels), positive=True)
        torch_shortcut_bn_weight = make_bn_tensor((1, 1, 1, out_channels))
        torch_shortcut_bn_bias = make_bn_tensor((1, 1, 1, out_channels))

        weights.shortcut_conv_weight = to_device_row_major(
            torch_shortcut_conv_weight, device=device, dtype=dtype
        )
        weights.shortcut_bn_running_mean = to_device_tile(
            torch_shortcut_bn_running_mean, device=device, dtype=dtype
        )
        weights.shortcut_bn_running_var = to_device_tile(
            torch_shortcut_bn_running_var, device=device, dtype=dtype
        )
        weights.shortcut_bn_weight = to_device_tile(
            torch_shortcut_bn_weight, device=device, dtype=dtype
        )
        weights.shortcut_bn_bias = to_device_tile(
            torch_shortcut_bn_bias, device=device, dtype=dtype
        )
        weights.shortcut_bias_tensor = None

    return weights


def create_resnet_layer_weights(
    *,
    layer_id,
    device,
    dtype,
    in_channels,
):
    spec = ResNetLayer.LAYER_SPECS[layer_id]
    out_channels = spec.out_channels
    num_blocks = spec.num_blocks
    first_stride = spec.first_stride

    weights = {}
    current_in_channels = in_channels

    for block_id in range(num_blocks):
        stride = first_stride if block_id == 0 else 1
        use_projection = (stride != 1) or (current_in_channels != out_channels)
        prefix = f"layer{layer_id}.{block_id}"

        torch_conv1_weight = torch.randn(
            (out_channels, current_in_channels, 3, 3),
            dtype=torch.bfloat16,
        )
        torch_conv2_weight = torch.randn(
            (out_channels, out_channels, 3, 3),
            dtype=torch.bfloat16,
        )

        weights[f"{prefix}.conv1.weight"] = to_device_row_major(
            torch_conv1_weight, device=device, dtype=dtype
        )
        weights[f"{prefix}.conv2.weight"] = to_device_row_major(
            torch_conv2_weight, device=device, dtype=dtype
        )

        for bn_name in ["bn1", "bn2"]:
            weights[f"{prefix}.{bn_name}.running_mean"] = to_device_tile(
                make_bn_tensor((1, 1, 1, out_channels)),
                device=device,
                dtype=dtype,
            )
            weights[f"{prefix}.{bn_name}.running_var"] = to_device_tile(
                make_bn_tensor((1, 1, 1, out_channels), positive=True),
                device=device,
                dtype=dtype,
            )
            weights[f"{prefix}.{bn_name}.weight"] = to_device_tile(
                make_bn_tensor((1, 1, 1, out_channels)),
                device=device,
                dtype=dtype,
            )
            weights[f"{prefix}.{bn_name}.bias"] = to_device_tile(
                make_bn_tensor((1, 1, 1, out_channels)),
                device=device,
                dtype=dtype,
            )

        if use_projection:
            torch_shortcut_conv_weight = torch.randn(
                (out_channels, current_in_channels, 1, 1),
                dtype=torch.bfloat16,
            )

            weights[f"{prefix}.shortcut.0.weight"] = to_device_row_major(
                torch_shortcut_conv_weight, device=device, dtype=dtype
            )
            weights[f"{prefix}.shortcut.1.running_mean"] = to_device_tile(
                make_bn_tensor((1, 1, 1, out_channels)),
                device=device,
                dtype=dtype,
            )
            weights[f"{prefix}.shortcut.1.running_var"] = to_device_tile(
                make_bn_tensor((1, 1, 1, out_channels), positive=True),
                device=device,
                dtype=dtype,
            )
            weights[f"{prefix}.shortcut.1.weight"] = to_device_tile(
                make_bn_tensor((1, 1, 1, out_channels)),
                device=device,
                dtype=dtype,
            )
            weights[f"{prefix}.shortcut.1.bias"] = to_device_tile(
                make_bn_tensor((1, 1, 1, out_channels)),
                device=device,
                dtype=dtype,
            )

        current_in_channels = out_channels

    return weights


def main():
    device = ttnn.open_device(
        device_id=0,
        l1_small_size=8192,
    )

    try:
        dtype = ttnn.bfloat16

        batch_size = 1
        input_height = 56
        input_width = 56
        in_channels = 64

        # Input tensor follows your constraint:
        # Shape([1, 1, 3136, 64]) where 3136 = 56 * 56
        torch_input = torch.randn(
            (batch_size, 1, input_height * input_width, in_channels),
            dtype=torch.bfloat16,
        )

        input_tensor = ttnn.from_torch(
            torch_input,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.create_sharded_memory_config(
                shape=(49, 64),
                core_grid=ttnn.CoreGrid(y=8, x=8),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
        )

        print("Input tensor:")
        print("  shape =", input_tensor.shape)
        print("  memory_config =", input_tensor.memory_config())
        print("  layout =", input_tensor.layout)
        print("  device =", input_tensor.device())

        # --------------------------------------------------
        # BasicBlock test: no projection
        # --------------------------------------------------
        basicblock_weights = create_basicblock_weights(
            device=device,
            dtype=dtype,
            in_channels=64,
            out_channels=64,
            use_projection=False,
        )

        basic_block = BasicBlock(
            weights=basicblock_weights,
            device=device,
            in_channels=64,
            out_channels=64,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            dtype=dtype,
            conv1_config=None,
            conv2_config=None,
            shortcut_conv_config=None,
        )

        basic_block_output = basic_block.forward(input_tensor)
        print("BasicBlock output shape =", basic_block_output.shape)

        # --------------------------------------------------
        # BasicBlock test: projection
        # --------------------------------------------------
        projection_weights = create_basicblock_weights(
            device=device,
            dtype=dtype,
            in_channels=64,
            out_channels=128,
            use_projection=True,
        )

        projection_block = BasicBlock(
            weights=projection_weights,
            device=device,
            in_channels=64,
            out_channels=128,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            stride=2,
            padding=1,
            dilation=1,
            groups=1,
            dtype=dtype,
            conv1_config=None,
            conv2_config=None,
            shortcut_conv_config=None,
        )

        projection_output = projection_block.forward(input_tensor)
        print("BasicBlock (projection) output shape =", projection_output.shape)

        # --------------------------------------------------
        # ResNetLayer test
        # layer2: 64 -> 128, first block stride=2
        # --------------------------------------------------
        layer_id = 2
        layer_weights = create_resnet_layer_weights(
            layer_id=layer_id,
            device=device,
            dtype=dtype,
            in_channels=64,
        )

        resnet_layer = ResNetLayer(
            layer_id=layer_id,
            weights=layer_weights,
            device=device,
            in_channels=64,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            dtype=dtype,
            conv2d_config=None,
        )

        layer_output = resnet_layer.forward(input_tensor)
        print("ResNetLayer output shape =", layer_output.shape)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()