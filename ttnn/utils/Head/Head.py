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
        # reshape về (N, H, W, C)
        x = ttnn.reshape(
            input_tensor,
            (
                self.batch_size,
                self.input_height,
                self.input_width,
                self.IN_FEATURES,
            ),
        )

        # global average pooling (manual via torch reference)
        x_torch = ttnn.to_torch(x).detach().cpu().float()
        x_torch = x_torch.mean(dim=(1, 2), keepdim=True)

        x = ttnn.from_torch(
            x_torch,
            dtype=x.dtype,
            layout=x.layout,
            device=x.device(),
            memory_config=self.memory_config,
        )

        # flatten
        x = ttnn.reshape(x, (self.batch_size, self.IN_FEATURES))

        # fully connected
        x = ttnn.linear(
            x,
            self.weights.fc_weight,
            bias=self.weights.fc_bias,
            memory_config=self.memory_config,
        )

        return x