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
python ../run.py --input_text "你好，请问你叫什么？" --max_output_len=50 --tokenizer_dir C:/Users/i/Documents/NV/DeepSeek-R1-Distill-Qwen-1.5B --engine_dir=./trt_engines-DS_1-5B_1gpu_fp16_wq_int4/
```

```
cd C:\Users\i\Documents\NV\TensorRT-LLM-0.17.0\benchmarks\python

python benchmark.py -m dec --engine_dir C:/Users/i/Documents/NV/TensorRT-LLM-0.17.0/examples/qwen/trt_engines-DS_1-5B_1gpu_fp16_wq_int4 --batch_size "1" --input_output_len "32,1;32,512;1024,1;1024,512" --quantization int4_weight_only --gpu_weights_percent 1
```
```
[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
[BENCHMARK] engine_dir trt_engines-DS_1-5B_1gpu_fp16_wq_int4 world_size 1 num_heads 12 num_kv_heads 2 num_layers 28 hidden_size 1536 vocab_size 151936 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 32 output_length 1 gpu_peak_mem(gb) 6.278 build_time(s) None tokens_per_sec 101.3 percentile95(ms) 11.656 percentile99(ms) 12.547 latency(ms) 9.872 compute_cap sm89 quantization QuantMode.INT4_WEIGHTS generation_time(ms) 0.029 total_generated_tokens 0.0 generation_tokens_per_second 0.0
[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
[BENCHMARK] engine_dir trt_engines-DS_1-5B_1gpu_fp16_wq_int4 world_size 1 num_heads 12 num_kv_heads 2 num_layers 28 hidden_size 1536 vocab_size 151936 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 32 output_length 512 gpu_peak_mem(gb) 6.28 build_time(s) None tokens_per_sec 144.68 percentile95(ms) 3667.135 percentile99(ms) 3667.135 latency(ms) 3538.792 compute_cap sm89 quantization QuantMode.INT4_WEIGHTS generation_time(ms) 3525.417 total_generated_tokens 511.0 generation_tokens_per_second 144.947
[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
[BENCHMARK] engine_dir trt_engines-DS_1-5B_1gpu_fp16_wq_int4 world_size 1 num_heads 12 num_kv_heads 2 num_layers 28 hidden_size 1536 vocab_size 151936 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 1024 output_length 1 gpu_peak_mem(gb) 6.29 build_time(s) None tokens_per_sec 13.47 percentile95(ms) 75.505 percentile99(ms) 76.35 latency(ms) 74.231 compute_cap sm89 quantization QuantMode.INT4_WEIGHTS generation_time(ms) 0.049 total_generated_tokens 0.0 generation_tokens_per_second 0.0
[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
[BENCHMARK] engine_dir trt_engines-DS_1-5B_1gpu_fp16_wq_int4 world_size 1 num_heads 12 num_kv_heads 2 num_layers 28 hidden_size 1536 vocab_size 151936 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 1024 output_length 512 gpu_peak_mem(gb) 6.301 build_time(s) None tokens_per_sec 135.8 percentile95(ms) 3845.297 percentile99(ms) 3845.297 latency(ms) 3770.31 compute_cap sm89 quantization QuantMode.INT4_WEIGHTS generation_time(ms) 3695.573 total_generated_tokens 511.0 generation_tokens_per_second 138.274

```

```
python benchmark.py -m dec --engine_dir C:/Users/i/Documents/NV/TensorRT-LLM-0.17.0/examples/qwen/trt_engines-DS_7B_1gpu_fp16_wq_int4 --batch_size "1" --input_output_len "32,1;32,512;1024,1;1024,512" --quantization int4_weight_only --gpu_weights_percent 1
```

```
[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
[BENCHMARK] engine_dir trt_engines-DS_7B_1gpu_fp16_wq_int4 world_size 1 num_heads 28 num_kv_heads 4 num_layers 28 hidden_size 3584 vocab_size 152064 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 32 output_length 1 gpu_peak_mem(gb) 10.264 build_time(s) None tokens_per_sec 42.68 percentile95(ms) 24.62 percentile99(ms) 25.499 latency(ms) 23.43 compute_cap sm89 quantization QuantMode.INT4_WEIGHTS generation_time(ms) 0.035 total_generated_tokens 0.0 generation_tokens_per_second 0.0
[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
[BENCHMARK] engine_dir trt_engines-DS_7B_1gpu_fp16_wq_int4 world_size 1 num_heads 28 num_kv_heads 4 num_layers 28 hidden_size 3584 vocab_size 152064 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 32 output_length 512 gpu_peak_mem(gb) 10.276 build_time(s) None tokens_per_sec 52.03 percentile95(ms) 9877.819 percentile99(ms) 9877.819 latency(ms) 9841.049 compute_cap sm89 quantization QuantMode.INT4_WEIGHTS generation_time(ms) 9815.935 total_generated_tokens 511.0 generation_tokens_per_second 52.058
[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
[BENCHMARK] engine_dir trt_engines-DS_7B_1gpu_fp16_wq_int4 world_size 1 num_heads 28 num_kv_heads 4 num_layers 28 hidden_size 3584 vocab_size 152064 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 1024 output_length 1 gpu_peak_mem(gb) 10.303 build_time(s) None tokens_per_sec 3.14 percentile95(ms) 319.098 percentile99(ms) 319.65 latency(ms) 318.066 compute_cap sm89 quantization QuantMode.INT4_WEIGHTS generation_time(ms) 0.047 total_generated_tokens 0.0 generation_tokens_per_second 0.0
[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
[BENCHMARK] engine_dir trt_engines-DS_7B_1gpu_fp16_wq_int4 world_size 1 num_heads 28 num_kv_heads 4 num_layers 28 hidden_size 3584 vocab_size 152064 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 1024 output_length 512 gpu_peak_mem(gb) 10.327 build_time(s) None tokens_per_sec 50.04 percentile95(ms) 10288.028 percentile99(ms) 10288.028 latency(ms) 10231.917 compute_cap sm89 quantization QuantMode.INT4_WEIGHTS generation_time(ms) 9913.705 total_generated_tokens 511.0 generation_tokens_per_second 51.545
```

========== GPTQ INT4
```
modelscope download --model tclf90/deepseek-r1-distill-qwen-7b-gptq-int4 --local_dir ./deepseek-r1-distill-qwen-7b-gptq-int4

cd C:\Users\intel\Documents\NV\TensorRT-LLM-0.17.0\examples\qwen

python convert_checkpoint.py --model_dir C:/Users/intel/Documents/NV/deepseek-r1-distill-qwen-7b-gptq-int4 --output_dir ./tllm_checkpoint_DS_7B_1gpu_gptq_int4 --dtype float16 --use_weight_only --weight_only_precision int4_gptq --per_group

trtllm-build --checkpoint_dir ./tllm_checkpoint_DS_7B_1gpu_gptq_int4 --output_dir ./trt_engines-DS_7B_1gpu_gptq_int4 --gemm_plugin float16

cd C:\Users\intel\Documents\NV\TensorRT-LLM-0.17.0\benchmarks\python

python benchmark.py -m dec --engine_dir C:/Users/intel/Documents/NV/TensorRT-LLM-0.17.0/examples/qwen/trt_engines-DS_7B_1gpu_gptq_int4 --batch_size "1" --input_output_len "32,1;32,512;1024,1;1024,512" --quantization int4_weight_only_gptq --gpu_weights_percent 1
```
```
[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
[BENCHMARK] engine_dir trt_engines-DS_7B_1gpu_gptq_int4 world_size 1 num_heads 28 num_kv_heads 4 num_layers 28 hidden_size 3584 vocab_size 152064 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 32 output_length 1 gpu_peak_mem(gb) 10.356 build_time(s) None tokens_per_sec 42.22 percentile95(ms) 24.532 percentile99(ms) 25.325 latency(ms) 23.686 compute_cap sm89 quantization QuantMode.PER_GROUP|INT4_WEIGHTS generation_time(ms) 0.039 total_generated_tokens 0.0 generation_tokens_per_second 0.0
[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
[BENCHMARK] engine_dir trt_engines-DS_7B_1gpu_gptq_int4 world_size 1 num_heads 28 num_kv_heads 4 num_layers 28 hidden_size 3584 vocab_size 152064 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 32 output_length 512 gpu_peak_mem(gb) 10.368 build_time(s) None tokens_per_sec 50.01 percentile95(ms) 10305.768 percentile99(ms) 10305.768 latency(ms) 10238.071 compute_cap sm89 quantization QuantMode.PER_GROUP|INT4_WEIGHTS generation_time(ms) 10213.736 total_generated_tokens 511.0 generation_tokens_per_second 50.031
[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
[BENCHMARK] engine_dir trt_engines-DS_7B_1gpu_gptq_int4 world_size 1 num_heads 28 num_kv_heads 4 num_layers 28 hidden_size 3584 vocab_size 152064 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 1024 output_length 1 gpu_peak_mem(gb) 10.395 build_time(s) None tokens_per_sec 3.09 percentile95(ms) 324.223 percentile99(ms) 325.163 latency(ms) 323.13 compute_cap sm89 quantization QuantMode.PER_GROUP|INT4_WEIGHTS generation_time(ms) 0.047 total_generated_tokens 0.0 generation_tokens_per_second 0.0
[TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
[BENCHMARK] engine_dir trt_engines-DS_7B_1gpu_gptq_int4 world_size 1 num_heads 28 num_kv_heads 4 num_layers 28 hidden_size 3584 vocab_size 152064 precision float16 batch_size 1 gpu_weights_percent 1.0 input_length 1024 output_length 512 gpu_peak_mem(gb) 10.419 build_time(s) None tokens_per_sec 47.7 percentile95(ms) 10776.559 percentile99(ms) 10776.559 latency(ms) 10733.078 compute_cap sm89 quantization QuantMode.PER_GROUP|INT4_WEIGHTS generation_time(ms) 10410.091 total_generated_tokens 511.0 generation_tokens_per_second 49.087
```
