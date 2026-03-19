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
        # input_tensor is [1, 1, B*H*W, C]
        x = ttnn.adaptive_avg_pool2d(
            input_tensor=input_tensor,
            batch_size=self.batch_size,
            input_h=self.input_height,
            input_w=self.input_width,
            channels=self.IN_FEATURES,
            output_size=[1, 1],
            memory_config=self.memory_config,
        )

        x = ttnn.reshape(x, (self.batch_size, self.IN_FEATURES))

        x = ttnn.linear(
            x,
            self.weights.fc_weight,
            bias=self.weights.fc_bias,
            memory_config=self.memory_config,
        )
        return x