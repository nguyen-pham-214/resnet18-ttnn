import ttnn

conv2d_config = {
    "conv0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        enable_act_double_buffer=True,
        reallocate_halo_output=True,

        # enable_activation_reuse=True,
        # force_split_reader=True,

        # act_block_h_override=32,
    ),

    "conv1.0.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,

        # deallocate_activation=True,

        # act_block_h_override=32,
        enable_act_double_buffer=False,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    "conv1.0.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,

        # deallocate_activation=True,

        # act_block_h_override=32,
        enable_act_double_buffer = False,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    "conv1.1.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,

        # deallocate_activation=True,

        # act_block_h_override=32,
        enable_act_double_buffer = False,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    "conv1.1.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,

        # deallocate_activation=True,

        # act_block_h_override=32,
        enable_act_double_buffer = False,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),

    "conv2.0.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        # enable_act_double_buffer=True,
        reallocate_halo_output=True,

        force_split_reader=True,

        # act_block_h_override=32,
    ),
    "conv2.0.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        # enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,

        # act_block_h_override=32,
    ),
    "conv2.0.shortcut": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        # enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,

        # act_block_h_override=32,
    ),
    "conv2.1.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        # enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,

        # act_block_h_override=32,
    ),
    "conv2.1.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        # enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,

        # act_block_h_override=32,
    ),

    "conv3.0.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,

        # act_block_h_override=32,
    ),
    "conv3.0.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,

        # act_block_h_override=32,
    ),
    "conv3.0.shortcut": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,

        # act_block_h_override=32,
    ),
    "conv3.1.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,

        # act_block_h_override=32,
    ),
    "conv3.1.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,

        # act_block_h_override=32,
    ),


    "conv4.0.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        # reshard_if_not_optimal=False,

        weights_dtype=ttnn.bfloat8_b,
        # override_sharding_config=True,  
        # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (6, 7))})
        enable_act_double_buffer=True,
        reallocate_halo_output=True,
    ),
    "conv4.0.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        # reshard_if_not_optimal=False,

        config_tensors_in_dram=True,
        weights_dtype=ttnn.bfloat8_b,
        # override_sharding_config=True,  
        # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (6, 7))})
        enable_act_double_buffer=True,
        reallocate_halo_output=True,
    ),
    
    "conv4.0.shortcut": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        # reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        enable_act_double_buffer=True,
        reallocate_halo_output=True,
    ),
    "conv4.1.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,

        # reshard_if_not_optimal=False,
        config_tensors_in_dram=True,
        weights_dtype=ttnn.bfloat8_b,
        enable_act_double_buffer=True,
        reallocate_halo_output=True,
    ),
    "conv4.1.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,

        # reshard_if_not_optimal=True,
        config_tensors_in_dram=True,
        weights_dtype=ttnn.bfloat8_b,
        enable_act_double_buffer=True,
        reallocate_halo_output=True,
    ),
}


