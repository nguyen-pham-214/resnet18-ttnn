import ttnn

conv2d_config = {
    "conv0": ttnn.Conv2dConfig(
        act_block_h_override=32,        
        
        act_block_w_div=1,          
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        config_tensors_in_dram=False,   
        core_grid=ttnn.CoreRangeSet({
            ttnn.CoreRange((0, 0), (6, 7))
        }),
        override_sharding_config=True,  
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        transpose_shards=False,      
        deallocate_activation=True,     
        enable_act_double_buffer=True, 
        enable_weights_double_buffer=False, 
        reallocate_halo_output=True,    
        reshard_if_not_optimal=False,  
        output_layout=ttnn.TILE_LAYOUT,
        full_inner_dim=False,           
        weights_dtype=ttnn.bfloat16,
    ),


}