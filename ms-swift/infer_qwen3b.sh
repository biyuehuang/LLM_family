CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/v0-20250313-110746/checkpoint-100 \
    --stream true \
    --merge_lora true \
    --temperature 0 \
    --max_model_len 8192 \
    --max_new_tokens 2048
