NV TRT_LLM on windows11

### 1. Download source code zip
https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v0.17.0


### 2. Install tensor-llm env
```
python3 -m venv trt-llm

trt-llm\Scripts\activate.bat

or 

conda create -n trt-llm python=3.10

conda activate trt-llm

pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124

pip install --extra-index-url https://pypi.nvidia.com/ tensorrt_llm==0.17.0.post1

[option]
pip install tensorrt_llm --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu121

[verify]
python -c "import tensorrt as trt; print(trt.__version__)"

python -c "import tensorrt_llm as trt_llm ; print(trt_llm.__version__)"
```

if meet fp4 error, modify fp4 to int4.
```
 File "C:\Users\i\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorrt_llm\_utils.py", line 223, in <module>
    nvfp4=trt.fp4)
```

### 3. Download models
```
pip install modelscope
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local_dir ./DeepSeek-R1-Distill-Qwen-1.5B
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local_dir ./DeepSeek-R1-Distill-Qwen-7B
```

### convert model to trt engine

========= DeepSeek-R1-Distill-Qwen-1.5B ============
```
cd C:\Users\i\Documents\NV\TensorRT-LLM-0.17.0\examples\qwen

python convert_checkpoint.py --model_dir C:/Users/i/Documents/NV/DeepSeek-R1-Distill-Qwen-1.5B --output_dir ./tllm_checkpoint_DS_1-5B_1gpu_fp16_wq_int4 --dtype float16 --use_weight_only --weight_only_precision int4

[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
0.17.0.post1
370it [00:28, 12.95it/s]
Total time of converting checkpoints: 00:00:31

trtllm-build --checkpoint_dir ./tllm_checkpoint_DS_1-5B_1gpu_fp16_wq_int4 --output_dir ./trt_engines-DS_1-5B_1gpu_fp16_wq_int4/ --gemm_plugin float16

[04/16/2025-09:39:37] [TRT-LLM] [I] Total time of building all engines: 00:00:35
```
inference
```
python ../run.py --input_text "你好，请问你叫什么？" --max_output_len=50 --tokenizer_dir C:/Users/i/Documents/NV/DeepSeek-R1-Distill-Qwen-1.5B --engine_dir=./trt_engines-1gpu_fp16_wq_int4/
```

```
cd C:\Users\i\Documents\NV\TensorRT-LLM-0.17.0\benchmarks\python

python benchmark.py -m dec --engine_dir C:/Users/i/Documents/NV/TensorRT-LLM-0.17.0/examples/qwen/trt_engines-DS_1-5B_1gpu_fp16_wq_int4 --batch_size "1" --input_output_len "32,512;1024,512" --quantization int4_weight_only --gpu_weights_percent 1

[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
Allocated 921.51 MiB for execution context memory.
C:\Users\i\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\nested\__init__.py:220: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\NestedTensorImpl.cpp:180.)
  return _nested.nested_tensor(
[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
[BENCHMARK] engine_dir trt_engines-1gpu_fp16_wq_int4 world_size 1 num_heads 12 num_kv_heads 2 num_layers 28 hidden_size 1536 vocab_size 151936 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 32 output_length 512 gpu_peak_mem(gb) 6.276 build_time(s) None tokens_per_sec 143.72 percentile95(ms) 3652.57 percentile99(ms) 3652.57 latency(ms) 3562.471 compute_cap sm89 quantization QuantMode.INT4_WEIGHTS generation_time(ms) 3549.948 total_generated_tokens 511.0 generation_tokens_per_second 143.946
[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
[BENCHMARK] engine_dir trt_engines-1gpu_fp16_wq_int4 world_size 1 num_heads 12 num_kv_heads 2 num_layers 28 hidden_size 1536 vocab_size 151936 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 1024 output_length 512 gpu_peak_mem(gb) 6.301 build_time(s) None tokens_per_sec 135.53 percentile95(ms) 3890.11 percentile99(ms) 3890.11 latency(ms) 3777.733 compute_cap sm89 quantization QuantMode.INT4_WEIGHTS generation_time(ms) 3703.125 total_generated_tokens 511.0 generation_tokens_per_second 137.992


python benchmark.py -m dec --engine_dir C:/Users/i/Documents/NV/TensorRT-LLM-0.17.0/examples/qwen/trt_engines-DS_7B_1gpu_fp16_wq_int4 --batch_size "1" --input_output_len "32,512;1024,512" --quantization int4_weight_only --gpu_weights_percent 1

[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
Allocated 1298.01 MiB for execution context memory.
C:\Users\i\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\nested\__init__.py:220: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\NestedTensorImpl.cpp:180.)
  return _nested.nested_tensor(
[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
[BENCHMARK] engine_dir trt_engines-DS_7B_1gpu_fp16_wq_int4 world_size 1 num_heads 28 num_kv_heads 4 num_layers 28 hidden_size 3584 vocab_size 152064 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 32 output_length 512 gpu_peak_mem(gb) 10.276 build_time(s) None tokens_per_sec 51.94 percentile95(ms) 9913.32 percentile99(ms) 9913.32 latency(ms) 9857.072 compute_cap sm89 quantization QuantMode.INT4_WEIGHTS generation_time(ms) 9832.157 total_generated_tokens 511.0 generation_tokens_per_second 51.972
[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
[BENCHMARK] engine_dir trt_engines-DS_7B_1gpu_fp16_wq_int4 world_size 1 num_heads 28 num_kv_heads 4 num_layers 28 hidden_size 3584 vocab_size 152064 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 1024 output_length 512 gpu_peak_mem(gb) 10.327 build_time(s) None tokens_per_sec 49.87 percentile95(ms) 10299.454 percentile99(ms) 10299.454 latency(ms) 10265.886 compute_cap sm89 quantization QuantMode.INT4_WEIGHTS generation_time(ms) 9948.403 total_generated_tokens 511.0 generation_tokens_per_second 51.365
```

========== GPTQ INT4
```
modelscope download --model tclf90/deepseek-r1-distill-qwen-7b-gptq-int4 --local_dir ./deepseek-r1-distill-qwen-7b-gptq-int4

cd C:\Users\i\Documents\NV\TensorRT-LLM-0.17.0\examples\qwen

python convert_checkpoint.py --model_dir C:/Users/i/Documents/NV/deepseek-r1-distill-qwen-7b-gptq-int4 --output_dir ./tllm_checkpoint_DS_7B_1gpu_gptq_int4 --dtype float16 --use_weight_only --weight_only_precision int4_gptq --per_group

trtllm-build --checkpoint_dir ./tllm_checkpoint_DS_7B_1gpu_gptq_int4 --output_dir ./trt_engines-DS_7B_1gpu_gptq_int4 --gemm_plugin float16

cd C:\Users\i\Documents\NV\TensorRT-LLM-0.17.0\benchmarks\python

python benchmark.py -m dec --engine_dir trt_engines-DS_7B_1gpu_gptq_int4 --batch_size "1" --input_output_len "32,512;1024,512" --quantization int4_weight_only --gpu_weights_percent 1
```
