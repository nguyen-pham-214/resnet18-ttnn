import torch
import ttnn

from InputStem import InputStem, InputStemWeights


def main():
    device = ttnn.open_device(
        device_id=0,
        l1_small_size=8192
    )

    try:
        batch_size = 1
        input_height = 224
        input_width = 224
        dtype = ttnn.bfloat16

        # Input: NHWC. 1 image with shape of (3,224,224)
        torch_input = torch.randn(
            (batch_size, input_height, input_width, 3),
            dtype=torch.bfloat16,
        )

        # Conv weight: [out_channels, in_channels, kH, kW]
        torch_conv_weight = torch.randn(
            (64, 3, 7, 7),
            dtype=torch.bfloat16,
        )

        # BN params: [out_channels]
        # BN params must be rank-4 for ttnn.batch_norm
        torch_bn_running_mean = torch.randn((1, 1, 1, 64), dtype=torch.bfloat16)
        torch_bn_running_var = torch.rand((1, 1, 1, 64), dtype=torch.bfloat16) + 1.0
        torch_bn_weight = torch.randn((1, 1, 1, 64), dtype=torch.bfloat16)
        torch_bn_bias = torch.randn((1, 1, 1, 64), dtype=torch.bfloat16)

        input_tensor = ttnn.from_torch(
            torch_input,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        conv_weight = ttnn.from_torch(
            torch_conv_weight,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        bn_running_mean = ttnn.from_torch(
            torch_bn_running_mean,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        bn_running_var = ttnn.from_torch(
            torch_bn_running_var,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        bn_weight = ttnn.from_torch(
            torch_bn_weight,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        bn_bias = ttnn.from_torch(
            torch_bn_bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        weights = InputStemWeights(
            conv_weight=conv_weight,
            bn_running_mean=bn_running_mean,
            bn_running_var=bn_running_var,
            bn_weight=bn_weight,
            bn_bias=bn_bias,
        )

        model = InputStem(
            weights=weights,
            device=device,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            dtype=dtype,
        )

        output = model.forward(input_tensor)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()



    