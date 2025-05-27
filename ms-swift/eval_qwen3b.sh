CUDA_VISIBLE_DEVICES=0 swift sample \
    --model output/v0-20250313-110746/checkpoint-100-merged \
    --sampler_engine pt \
    --num_return_sequences 1 \
    --dataset dataset/lightv1 \
