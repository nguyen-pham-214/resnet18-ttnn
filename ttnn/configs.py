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

        # act_block_h_override=64,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),

    "conv1.0.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,

        # deallocate_activation=True,

        # act_block_h_override=64,
        enable_act_double_buffer=False,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    "conv1.0.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,

        # deallocate_activation=True,

        # act_block_h_override=64,
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

        # act_block_h_override=64,
        enable_act_double_buffer = False,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    "conv1.1.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,

        # deallocate_activation=True,

        # act_block_h_override=64,
        enable_act_double_buffer = False,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),

    "conv2.0.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,
        # deallocate_activation=True,

        # act_block_h_override=64,
        enable_act_double_buffer=True,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    "conv2.0.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,
        # deallocate_activation=True,
        
        # act_block_h_override=64,
        enable_act_double_buffer=True,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    "conv2.0.shortcut": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,
        # deallocate_activation=True,
        
        # act_block_h_override=64,
        enable_act_double_buffer=True,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    "conv2.1.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,
        # deallocate_activation=True,
        
        # act_block_h_override=64,
        enable_act_double_buffer=True,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    "conv2.1.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,
        # deallocate_activation=True,
        
        # act_block_h_override=64,
        enable_act_double_buffer=True,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),

    "conv3.0.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,
        # deallocate_activation=True,
        
        # act_block_h_override=64,
        enable_act_double_buffer=True,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    "conv3.0.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,
        # deallocate_activation=True,
        
        # act_block_h_override=64,
        enable_act_double_buffer=True,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    "conv3.0.shortcut": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,
        # deallocate_activation=True,
        
        # act_block_h_override=64,
        enable_act_double_buffer=True,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    "conv3.1.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,
        # deallocate_activation=True,
        
        # act_block_h_override=64,
        enable_act_double_buffer=True,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    "conv3.1.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,

        reshard_if_not_optimal=False,
        weights_dtype=ttnn.bfloat8_b,
        reallocate_halo_output=True,
        # deallocate_activation=True,
        
        # act_block_h_override=64,
        enable_act_double_buffer=True,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),


    "conv4.0.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        reshard_if_not_optimal=False,

        weights_dtype=ttnn.bfloat8_b,
        enable_act_double_buffer=True,
        reallocate_halo_output=True,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    "conv4.0.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        reshard_if_not_optimal=False,

        weights_dtype=ttnn.bfloat8_b,
        enable_act_double_buffer=True,
        reallocate_halo_output=True,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    
    "conv4.0.shortcut": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        reshard_if_not_optimal=False,

        weights_dtype=ttnn.bfloat8_b,
        enable_act_double_buffer=True,
        reallocate_halo_output=True,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    "conv4.1.0": ttnn.Conv2dConfig(
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        reshard_if_not_optimal=False,

        weights_dtype=ttnn.bfloat8_b,
        enable_act_double_buffer=True,
        reallocate_halo_output=True,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
    "conv4.1.1": ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        reshard_if_not_optimal=False,

        weights_dtype=ttnn.bfloat8_b,
        enable_act_double_buffer=True,
        reallocate_halo_output=True,
        config_tensors_in_dram = True   # moves tiny config tensors, not main activations
    ),
}


