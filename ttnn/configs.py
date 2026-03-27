import ttnn

conv2d_config = {
    # stem : (1, 224, 224, 3) 
    "conv0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat16,
        # enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        enable_activation_reuse=True,
        force_split_reader=True,

        # act_block_h_override=32 * 7,
    ),

    # layer 1 : (1, 1, 50176, 64)
    "conv1.0.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat16,
        # enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,
    ),
    "conv1.0.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat16,
        # enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,
    ),
    "conv1.1.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat16,
        # enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,
    ),
    "conv1.1.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat16,
        # enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,
    ),

    # layer 2 : (1, 1, 50176, 64) 
    "conv2.0.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat16,
        # enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,
    ),
    "conv2.0.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat16,
        # enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,
    ),
    "conv2.0.shortcut": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat16,
        # enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,
    ),
    "conv2.1.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat16,
        # enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,
    ),
    "conv2.1.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat16,
        # enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,
    ),

    # layer 3 : (1, 1, 12544, 128) 
    "conv3.0.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat16,
        enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,
    ),
    "conv3.0.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat16,
        enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,
    ),
    "conv3.0.shortcut": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat16,
        enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,
    ),
    "conv3.1.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat16,
        enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,
    ),
    "conv3.1.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat16,
        enable_act_double_buffer=True,
        # reallocate_halo_output=True,

        force_split_reader=True,
    ),


    # layer 4 : (1, 1, 3136, 256) 
    "conv4.0.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        # reshard_if_not_optimal=False,

        weights_dtype=ttnn.bfloat16,
        # override_sharding_config=True,  
        # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (6, 7))})
        enable_act_double_buffer=True,
        reallocate_halo_output=True,
    ),
    "conv4.0.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        # reshard_if_not_optimal=False,

        config_tensors_in_dram=True,
        weights_dtype=ttnn.bfloat16,
        # override_sharding_config=True,  
        # core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (6, 7))})
        enable_act_double_buffer=True,
        reallocate_halo_output=True,
    ),
    
    "conv4.0.shortcut": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        # reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat16,
        enable_act_double_buffer=True,
        reallocate_halo_output=True,
    ),
    "conv4.1.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,

        # reshard_if_not_optimal=False,
        config_tensors_in_dram=True,
        weights_dtype=ttnn.bfloat16,
        enable_act_double_buffer=True,
        reallocate_halo_output=True,
    ),
    "conv4.1.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,

        # reshard_if_not_optimal=True,
        config_tensors_in_dram=True,
        weights_dtype=ttnn.bfloat16,
        enable_act_double_buffer=True,
        reallocate_halo_output=True,
    ),
}


