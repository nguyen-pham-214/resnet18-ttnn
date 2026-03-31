from dataclasses import dataclass
from typing import Optional

import ttnn


@dataclass
class HeadWeights:
    fc_weight: ttnn.Tensor
    fc_bias: Optional[ttnn.Tensor] = None


class ResNetHead:
    def __init__(
        self,
        *,
        weights: HeadWeights,
        batch_size: int,
        in_features: int = 512,
        num_classes: int = 1000,
        input_height: int = 7,
        input_width: int = 7,
        dtype=ttnn.bfloat16,
        memory_config=None,
    ) -> None:
        self.weights = weights
        self.batch_size = batch_size
        self.in_features = in_features
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.dtype = dtype

    def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        shape = [int(d) for d in input_tensor.shape]
        expected = self.batch_size * self.input_height * self.input_width

        # [1,1,B*H*W,C] -> [B,H,W,C]
        if shape == [1, 1, expected, self.in_features]:
            x = ttnn.reshape(
                input_tensor,
                (self.batch_size, self.input_height, self.input_width, self.in_features),
            )
        else:
            x = input_tensor

        # flatten spatial -> [B, H*W, C]
        x = ttnn.reshape(
            x,
            (self.batch_size, self.input_height * self.input_width, self.in_features),
        )

        # manual global average: mean over dim=1
        x = ttnn.mean(x, dim=1)  # -> [B, C]

        # FC
        x = ttnn.linear(
            x,
            self.weights.fc_weight,
            bias=self.weights.fc_bias,
        )

        out_shape = [int(d) for d in x.shape]

        if len(out_shape) == 4:
            _, h, w, c = out_shape
        elif len(out_shape) == 3:
            h, w, c = out_shape
        elif len(out_shape) == 2:
            _, c = out_shape
            h, w = 1, 1
        elif len(out_shape) == 1:
            c = out_shape[0]
            h, w = 1, 1
        else:
            raise RuntimeError(f"Unsupported final output rank: shape={x.shape}")

        self.final_dimension = (c, h, w)

        return x