import ttnn

conv2d_config = {
    "conv0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
    ),

    # layer 1
    "conv1.0.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
    ),
    "conv1.0.1": ttnn.Conv2dConfig(
    ),
    "conv1.1.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
    ),
    "conv1.1.1": ttnn.Conv2dConfig(
    ),

    # layer 2
    "conv2.0.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
    ),
    "conv2.0.1": ttnn.Conv2dConfig(
    ),
    "conv2.0.shortcut": ttnn.Conv2dConfig(
    ),
    "conv2.1.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
    ),
    "conv2.1.1": ttnn.Conv2dConfig(
    ),

    # layer 3
    "conv3.0.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
    ),
    "conv3.0.1": ttnn.Conv2dConfig(
    ),
    "conv3.0.shortcut": ttnn.Conv2dConfig(
    ),
    "conv3.1.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
    ),
    "conv3.1.1": ttnn.Conv2dConfig(
    ),


    # layer 4
    "conv4.0.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
    ),
    "conv4.0.1": ttnn.Conv2dConfig(
    ),
    "conv4.0.shortcut": ttnn.Conv2dConfig(
    ),
    "conv4.1.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
    ),
    "conv4.1.1": ttnn.Conv2dConfig(
    ),
}