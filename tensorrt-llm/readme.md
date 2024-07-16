https://github.com/NVIDIA/TensorRT-LLM
https://nvidia.github.io/TensorRT-LLM/installation/windows.html
https://developer.download.nvidia.cn/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz

### Install
```
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs wget python-is-python3 vim
pip install tensorrt_llm==0.9.0 -U --extra-index-url https://pypi.nvidia.com
pip install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com
```
### Verify
```
python -c "import tensorrt_llm"
python -c "import tensorrt_llm; print(tensorrt_llm._utils.trt_version())"
```
### Build
```
python3 ./scripts/build_wheel.py --clean  --trt_root /usr/local/tensorrt
```

### Fix
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12/compat

### Docker
--shm-size=10.24gb
```
docker run --runtime=nvidia --gpus all --shm-size=10.24gb -v /opt:/opt -v /data:/data -e http_proxy='http://child-prc.intel.com:913' -e https_proxy='http://child-prc.intel.com:913' --entrypoint /bin/bash -it nvidia/cuda:12.5.0-devel-ubuntu22.04 

docker run --runtime=nvidia --gpus all -v /opt/WD:/opt/WD -e http_proxy='http://child-prc.intel.com:913' -e https_proxy='http://child-prc.intel.com:913' --entrypoint /bin/bash -it nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3

docker run --runtime=nvidia --gpus all -v /opt:/opt -v /data:/data -e http_proxy='http://child-prc.intel.com:913' -e https_proxy='http://child-prc.intel.com:913' --entrypoint /bin/bash -it tensorrt_llm/devel

docker start flamboyant_easley
docker exec -it flamboyant_easley /bin/bash
python3 ./scripts/build_wheel.py --clean  --trt_root /usr/local/tensorrt --benchmarks
```

### Convert
```
#chatglm3-6b
python3 examples/chatglm/convert_checkpoint.py --model_dir /data/chatglm3-6b \
	--output_dir /data/trt_ckpt/chatglm3_6b/fp16/1-gpu 

trtllm-build --checkpoint_dir /data/trt_ckpt/chatglm3_6b/fp16/1-gpu \
  --output_dir /data/trt_engines/chatglm3_6b/fp16/1-gpu \
  --gemm_plugin float16 --max_batch_size 1 --max_input_len 8192
```  
INT8
```
python3 examples/chatglm/convert_checkpoint.py --model_dir /data/chatglm3-6b \
	--output_dir /data/trt_ckpt/chatglm3_6b/int8/1-gpu --use_weight_only --weight_only_precision int8

trtllm-build --checkpoint_dir /data/trt_ckpt/chatglm3_6b/int8/1-gpu \
  --output_dir /data/trt_engines/chatglm3_6b/int8/1-gpu \
  --gemm_plugin float16 --max_batch_size 32
```
INT4
```
python3 examples/chatglm/convert_checkpoint.py --model_dir /data/chatglm3-6b \
	--output_dir /data/trt_ckpt/chatglm3_6b/int4/1-gpu --use_weight_only --weight_only_precision int4

trtllm-build --checkpoint_dir /data/trt_ckpt/chatglm3_6b/int4/1-gpu \
  --output_dir /data/trt_engines/chatglm3_6b/int4/1-gpu \
  --gemm_plugin float16 --max_batch_size 32
```  
### Llama2-7B
```
#FP16
python convert_checkpoint.py --model_dir /opt/WD/009-models/models/Llama-2-7B-Chat-hf --output_dir /opt/WD/009-models/trt_ckpt/llama/tllm_checkpoint_1gpu_fp16
trtllm-build --checkpoint_dir /opt/WD/009-models/trt_ckpt/llama/tllm_checkpoint_1gpu_fp16 --output_dir /opt/WD/009-models/trt_engines/llama/7B/fp16/1-gpu \
  --gemm_plugin auto --max_batch_size 16

#INT8
python convert_checkpoint.py --model_dir /opt/WD/009-models/models/Llama-2-7B-Chat-hf --output_dir /opt/WD/009-models/trt_ckpt/llama/tllm_checkpoint_1gpu_int8 \
  --use_weight_only --weight_only_precision int8 --load_model_on_cpu 
trtllm-build --checkpoint_dir /opt/WD/009-models/trt_ckpt/llama/tllm_checkpoint_1gpu_int8 --output_dir /opt/WD/009-models/trt_engines/llama/7B/int8/1-gpu \
  --gemm_plugin auto --max_batch_size 16

#INT4
python convert_checkpoint.py --model_dir /opt/WD/009-models/models/Llama-2-7B-Chat-hf --output_dir /opt/WD/009-models/trt_ckpt/llama/tllm_checkpoint_1gpu_int4 \
  --use_weight_only --weight_only_precision int4 --load_model_on_cpu 
trtllm-build --checkpoint_dir /opt/WD/009-models/trt_ckpt/llama/tllm_checkpoint_1gpu_int4 --output_dir /opt/WD/009-models/trt_engines/llama/7B/int4/1-gpu \
  --gemm_plugin auto --max_batch_size 16
```
  
### llama3
```
#FP16
python convert_checkpoint.py --model_dir ~/Meta-Llama-3-8B-Instruct --output_dir ~/trt_ckpt/llama3/tllm_checkpoint_1gpu_fp16 \
  --dtype float16 --tp_size 1
trtllm-build --checkpoint_dir ~/trt_ckpt/llama3/tllm_checkpoint_1gpu_fp16 --output_dir ~/trt_engines/llama3/7B/fp16/1-gpu \
  --gemm_plugin auto --max_batch_size 1
FP8
python ../quantization/quantize.py --model_dir ~/Meta-Llama-3-8B-Instruct \
                                   --dtype float16 --qformat fp8 --kv_cache_dtype fp8 \
                                   --output_dir ~/trt_ckpt/llama3/tllm_checkpoint_1gpu_fp8 \
                                   --calib_size 512 --tp_size 1
trtllm-build --checkpoint_dir ~/trt_ckpt/llama3/tllm_checkpoint_1gpu_fp8 --output_dir ~/trt_engines/llama3/7B/fp8/1-gpu \
  --gemm_plugin auto --max_batch_size 1
```
  
### Qwen
```
#INT8
python convert_checkpoint.py --model_dir /opt/WD/009-models/models/Qwen1.5-7B-Chat \
  --output_dir /opt/WD/009-models/trt_ckpt/qwen/tllm_checkpoint_1gpu_int8 --qwen_type qwen2 --dtype float16 --use_weight_only --weight_only_precision int8

trtllm-build --checkpoint_dir /opt/WD/009-models/trt_ckpt/qwen/tllm_checkpoint_1gpu_int8 \
  --output_dir /opt/WD/009-models/trt_engines/qwen/7B/int8/1-gpu --gemm_plugin float16 --max_batch_size 4 --max_input_len 4096

#INT4
python convert_checkpoint.py --model_dir /opt/WD/009-models/models/Qwen1.5-7B-Chat \
  --output_dir /opt/WD/009-models/trt_ckpt/qwen/tllm_checkpoint_1gpu_int4 --qwen_type qwen2 --dtype float16 --use_weight_only --weight_only_precision int4

trtllm-build --checkpoint_dir /opt/WD/009-models/trt_ckpt/qwen/tllm_checkpoint_1gpu_int4 \
  --output_dir /opt/WD/009-models/trt_engines/qwen/7B/int4/1-gpu --gemm_plugin float16 --max_batch_size 4 --max_input_len 4096

#FP8
python ../quantization/quantize.py --model_dir ~/Qwen1.5-7B-Chat --dtype float16 --qformat fp8 \
  --kv_cache_dtype fp8 --output_dir ~/trt_ckpt/qwen/tllm_checkpoint_1gpu_fp8 --calib_size 512

trtllm-build --checkpoint_dir ~/trt_ckpt/qwen/tllm_checkpoint_1gpu_fp8 --output_dir ~/trt_engines/qwen/7B/fp8/1-gpu \
  --gemm_plugin float16 --max_batch_size 1 --max_input_len 8192
```
             
### Benchmark
```
#chatglm3
python benchmarks/python/benchmark.py -m chatglm3_6b --batch_size "1;4;8;16;32"  --input_output_len "1024,512;2048,512"

python benchmarks/python/benchmark.py -m chatglm3_6b --engine_dir /data/trt_engines/chatglm3_6b/fp16/1-gpu --batch_size "1;4;8;16;32"  --input_output_len "1024,1;1024,512"
python benchmarks/python/benchmark.py -m chatglm3_6b --engine_dir /data/trt_engines/chatglm3_6b/int8/1-gpu --batch_size "1;4;8;16;32"  --input_output_len "1024,1;1024,512"
python benchmarks/python/benchmark.py -m chatglm3_6b --engine_dir /data/trt_engines/chatglm3_6b/int4/1-gpu --batch_size "1;4;8;16;32"  --input_output_len "1024,1;1024,512"

#Llama2-7B
python benchmark.py -m llama_7b --batch_size "1;4;8;16"  --input_output_len "1024,1;1024,512"
python benchmark.py -m llama_7b --batch_size "1;4;8;16"  --input_output_len "1024,512"

python benchmark.py -m llama_7b --engine_dir /opt/WD/009-models/trt_engines/llama/7B/int8/1-gpu --batch_size "1;4;8"  --input_output_len "1024,1"
python benchmark.py -m llama_7b --engine_dir /opt/WD/009-models/trt_engines/llama/7B/int8/1-gpu --batch_size "1;4;8"  --input_output_len "1024,512"

python benchmark.py -m llama_7b --engine_dir /opt/WD/009-models/trt_engines/llama/7B/int4/1-gpu --batch_size "1;4;8"  --input_output_len "1024,1"
python benchmark.py -m llama_7b --engine_dir /opt/WD/009-models/trt_engines/llama/7B/int4/1-gpu --batch_size "1;4;8"  --input_output_len "1024,512"

#llama3
python benchmark.py -m llama_7b --engine_dir ~/trt_engines/llama3/7B/fp16/1-gpu --batch_size "1"  --input_output_len "1024,512"
python benchmark.py -m llama_7b --engine_dir ~/trt_engines/llama3/7B/fp16/1-gpu --batch_size "4"  --input_output_len "1024,512"
python benchmark.py -m llama_7b --engine_dir ~/trt_engines/llama3/7B/fp16/1-gpu --batch_size "8"  --input_output_len "1024,512"

#Qwen
python benchmark.py -m qwen1.5_7b_chat --batch_size "1" --max_batch_size 1 --input_output_len "1024,1;1024,512;2048,1;2048,512;4096,1;4096,512;8192,1;8192,512"
python benchmark.py -m qwen1.5_7b_chat --batch_size "4" --max_batch_size 4 --max_input_len 4096 --input_output_len "1024,1;1024,512;2048,1;2048,512;4096,1;4096,512"
python benchmark.py -m qwen1.5_7b_chat --batch_size "8" --max_batch_size 8 --max_input_len 2048 --input_output_len "1024,1;1024,512;2048,1;2048,512"

python benchmark.py -m qwen1.5_7b_chat --engine_dir /opt/WD/009-models/trt_engines/qwen/7B/int8/1-gpu \
  --batch_size "1" --max_batch_size 1 --max_input_len 8192 --input_output_len "1024,1;1024,512;2048,1;2048,512;4096,1;4096,512;8192,1;8192,512"
python benchmark.py -m qwen1.5_7b_chat --engine_dir /opt/WD/009-models/trt_engines/qwen/7B/int8/1-gpu \
  --batch_size "4" --max_batch_size 4 --max_input_len 2048 --input_output_len "1024,1;1024,512;2048,1;2048,512"
python benchmark.py -m qwen1.5_7b_chat --engine_dir /opt/WD/009-models/trt_engines/qwen/7B/int8/1-gpu \
  --batch_size "8" --max_batch_size 8 --max_input_len 1024 --input_output_len "1024,1;1024,512"

python benchmark.py -m qwen1.5_7b_chat --engine_dir /opt/WD/009-models/trt_engines/qwen/7B/int4/1-gpu \
  --batch_size "1" --max_batch_size 1 --max_input_len 8192 --input_output_len "1024,1;1024,512;2048,1;2048,512;4096,1;4096,512;8192,1;8192,512"
python benchmark.py -m qwen1.5_7b_chat --engine_dir /opt/WD/009-models/trt_engines/qwen/7B/int4/1-gpu \
  --batch_size "4" --max_batch_size 4 --max_input_len 2048 --input_output_len "1024,1;1024,512;2048,1;2048,512"
python benchmark.py -m qwen1.5_7b_chat --engine_dir /opt/WD/009-models/trt_engines/qwen/7B/int4/1-gpu \
  --batch_size "8" --max_batch_size 8 --max_input_len 1024 --input_output_len "1024,1;1024,512"
  
#Llama2-13B
python3 examples/llama/convert_checkpoint.py --model_dir Llama2-13B-Chat-hf/ --output_dir /ckpt/Llama2/13B/tllm_checkpoint_gpu_int4 --dtype float16 \
  --use_weight_only --weight_only_precision int4 --tp_size 2
trtllm-build --checkpoint_dir /ckpt/Llama2/13B/tllm_checkpoint_gpu_int4 --output_dir /trt_engines/llama2_13b/int4/2-gpu --gemm_plugin float16 \
  --max_batch_size 1 --max_input_len 2048 --max_output_len 512 --tp_size 2 --use_custom_all_reduce disable
mpirun -n 2 --allow-run-as-root python3 benchmarks/python/benchmark.py -m llama_13b --engine_dir /trt_engines/llama2_13b/int4/2-gpu/  --batch_size 1 --input_output_len "2048,512"

#fp8
python ../quantization/quantize.py --model_dir /TensorRT-LLM/Llama2-13B-Chat-hf/ \
  --dtype float16 --qformat fp8 --kv_cache_dtype fp8 \
  --output_dir /ckpt/llama3/tllm_checkpoint_2gpu_fp8 \
  --calib_size 512 --tp_size 2
trtllm-build --checkpoint_dir /ckpt/llama3/tllm_checkpoint_2gpu_fp8 --output_dir /trt_engines/llama2_13b/fp8/2-gpu --gemm_plugin float16 \
  --max_batch_size 1 --max_input_len 1024 --max_output_len 512 --tp_size 2 --use_custom_all_reduce disable --workers 2
mpirun -n 2 --allow-run-as-root python3 ../../benchmarks/python/benchmark.py -m llama_13b --engine_dir /trt_engines/llama2_13b/fp8/2-gpu/  --batch_size 1 --input_output_len "1024,512"

#Qwen1.5-14B
python ../quantization/quantize.py --model_dir /TensorRT-LLM/Qwen1.5-32B/ \
  --dtype float16 --qformat fp8 --kv_cache_dtype fp8 \
  --output_dir /ckpt/qwen/tllm_checkpoint_2gpu_fp8 \
  --calib_size 512 --tp_size 2
trtllm-build --checkpoint_dir /ckpt/qwen/tllm_checkpoint_2gpu_fp8 --output_dir /trt_engines/qwen/fp8/2-gpu --gemm_plugin float16 \
  --max_batch_size 1 --max_input_len 1024 --max_output_len 512 --tp_size 2 --use_custom_all_reduce disable --workers 2
mpirun -n 2 --allow-run-as-root python3 ../../benchmarks/python/benchmark.py -m llama_13b --engine_dir /trt_engines/llama2_13b/fp8/2-gpu/  --batch_size 1 --input_output_len "1024,512"

python3 examples/qwen/convert_checkpoint.py --model_dir Qwen1.5-32B/ --output_dir /ckpt/Qwen/32B/tllm_checkpoint_gpu_int4 --dtype float16 \
  --use_weight_only --weight_only_precision int4 --tp_size 2 --load_model_on_cpu --qwen_type qwen2
trtllm-build --checkpoint_dir /ckpt/Qwen/32B/tllm_checkpoint_gpu_int4/ --output_dir /trt_engines/Qwen/32B/int4/2-gpu --gemm_plugin float16 \
--max_num_tokens 2048 --max_batch_size 1  --tp_size 2 --use_custom_all_reduce disable --workers 2 --max_input_len 1024 --max_output_len 512
mpirun -n 2 --allow-run-as-root python3 benchmarks/python/benchmark.py -m qwen1.5_7b_chat --engine_dir /trt_engines/Qwen/32B/int4/2-gpu/ --batch_size 1 --input_output_len "2048,512"


trtllm-build --checkpoint_dir /ckpt/Qwen/32B/tllm_checkpoint_gpu_int8/ --output_dir /trt_engines/Qwen/32B/int8/2-gpu --gemm_plugin float16 \
--max_num_tokens 16384 --max_batch_size 8  --tp_size 2 --use_custom_all_reduce disable --workers 2 --max_input_len 2048 --max_output_len 512
mpirun -n 2 --allow-run-as-root python3 benchmarks/python/benchmark.py -m qwen1.5_7b_chat --engine_dir /trt_engines/Qwen/32B/int8/2-gpu/ --batch_size 8 --input_output_len "1024,512"
mpirun -n 2 --allow-run-as-root python3 benchmarks/python/benchmark.py -m qwen1.5_7b_chat --engine_dir /trt_engines/Qwen/32B/int8/2-gpu/ --batch_size 8 --input_output_len "2048,512"


python examples/summarize.py --test_trt_llm \
                       --hf_model_dir Qwen1.5-32B \
                       --data_type fp16 \
                       --engine_dir /trt_engines/Qwen/32B/int4/2-gpu \
                       --max_input_length 2048 \
                       --output_len 512


python -m vllm.entrypoints.openai.api_server --model /data/Qwen1.5-14B-Chat/ --trust-remote-code --tensor-parallel-size 2 --disable-custom-all-reduce --dtype float16 --max-model-len 14432 --max-num-seqs 4
python benchmark_serving.py --model /data/Qwen1.5-14B-Chat --dataset ShareGPT_V3_unfiltered_cleaned_split.json --trust-remote-code --backend vllm
```
