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
        self.memory_config = memory_config

    def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        x = input_tensor

        x = ttnn.adaptive_avg_pool2d(
            x,
            batch_size=self.batch_size,
            input_h=self.input_height,
            input_w=self.input_width,
            channels=self.in_features,
            output_size=[1, 1],
        )

        # pooled output -> feed FC
        x = ttnn.reshape(x, (self.batch_size, self.in_features))
        
        # remove sharding, but stay in L1
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        # now tilize
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        x = ttnn.linear(x, self.weights.fc_weight, bias=self.weights.fc_bias)

        return x