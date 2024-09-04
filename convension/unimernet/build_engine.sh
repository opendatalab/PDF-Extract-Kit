
BATCHSIZE=128
PARTITION=AI4Chem
srun -p $PARTITION -N1 -c8 --gres=gpu:1 --mpi=pmi2  trtllm-build --checkpoint_dir examples/multimodal/trt_engines/unimernet/bfloat16/decoder \
    --output_dir examples/multimodal/trt_engines.b$BATCHSIZE/unimernet/1-gpu/bfloat16/decoder     --paged_kv_cache disable \
    --moe_plugin disable     --enable_xqa disable     --gemm_plugin bfloat16     --bert_attention_plugin bfloat16 \
    --gpt_attention_plugin bfloat16     --remove_input_padding enable     --max_beam_width 1     --max_batch_size $BATCHSIZE \
    --max_seq_len 101     --max_input_len 1     --max_encoder_input_len 588

srun -p $PARTITION -N1 -c8 --gres=gpu:1 --mpi=pmi2 python build_visual_engine.py $BATCHSIZE "examples/multimodal/trt_engines.b$BATCHSIZE/vision_encoder/"