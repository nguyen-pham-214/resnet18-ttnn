from dataclasses import dataclass
from typing import Optional

import ttnn


@dataclass
class HeadWeights:
    fc_weight: ttnn.Tensor
    fc_bias: Optional[ttnn.Tensor] = None


class ResNetHead:
    IN_FEATURES = 512

    def __init__(
        self,
        *,
        weights: HeadWeights,
        batch_size: int,
        num_classes: int,
        input_height: int,
        input_width: int,
        memory_config=None,
    ) -> None:
        self.weights = weights
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.memory_config = memory_config


    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.avg_pool2d(
            input_tensor,
            batch_size=self.batch_size,
            input_h=self.input_height,
            input_w=self.input_width,
            channels=self.IN_FEATURES,
            kernel_size=[self.input_height, self.input_width],
            stride=[1, 1],
            padding=[0, 0],
            memory_config=self.memory_config,
            output_layout=ttnn.TILE_LAYOUT,
        )

        # flatten pooled output: [1,1,N, C] -> [N, C]
        x = ttnn.reshape(x, (self.batch_size, self.IN_FEATURES))

        x = ttnn.linear(
            x,
            self.weights.fc_weight,
            bias=self.weights.fc_bias,
            memory_config=self.memory_config,
        )
        return x