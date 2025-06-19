#!/bin/bash
# 定义要测试的 num_prompt_values 和 random-input-len 值
# 定义要测试的 model_name 和 gpu_num 值
num_prompt_values=(2 1 2 4 6 8 10 12 14 16 18 20 22 24)
random_input_len=1024
random_output_len=512
model_name="DeepSeek-R1-Distill-Llama-70B"
gpu_num=4
# 基础命令
base_command="python /llm/vllm/benchmarks/benchmark_serving.py \
        --model /llm/models/${model_name} \
        --served-model-name ${model_name} \
        --dataset-name random \
        --trust_remote_code \
        --port 8004 \
        --ignore-eos \
        --random-output-len=${random_output_len}"
log_file="${model_name}-${random_input_len}-gpu${gpu_num}-g95-2000-3000.log"
# 清空或创建日志文件
> "$log_file"
for num_prompt in "${num_prompt_values[@]}"
do
        echo "Running benchmark with num_prompt=${num_prompt}, random-input-len=${random_input_len}..." | tee -a "$log_file"
        $base_command --num_prompt "$num_prompt" --random-input-len "$random_input_len" | tee -a "$log_file" 2>&1
        echo "Completed benchmark with num_prompt=${num_prompt}, random-input-len=${random_input_len}." | tee -a "$log_file"
        echo "------------------------------------------------------------" | tee -a "$log_file"
done
echo "All benchmarks completed. Results are saved in $log_file"
