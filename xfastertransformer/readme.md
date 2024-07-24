安装环境
```
docker pull intel/xfastertransformer:1.7.0
docker run -it --name xfastertransformer_1_7_0 --privileged --shm-size=16g -v /home/model:/data -e "http_proxy=$http_proxy" -e "https_proxy=$https_proxy" intel/xfastertransformer:1.7.0
```
首次运行容器用
```
docker run
```

之后通过  
```
docker ps ## 获得“docker id”
docker start “docker id”
docker exec -it “docker id” bash
```
模型转换指令
```
python -c 'import xfastertransformer as xft; xft.LlamaConvert().convert("${HF_DATASET_DIR}","${OUTPUT_DIR}")'
```

运行模型推理，BF16精度
```
cd /root/xFasterTransformer/benchmark
bash run_benchmark.sh -s 1 -bs 1 -d bf16 -kvd fp16 -m llama-2-7b -mp /data/Llama-2-7B-Chat-hf-xft/ -tp /data/Llama-2-7B-Chat-hf/ -in 1024 -out 512 -i 1
```

FP16精度
```
bash run_benchmark.sh -s 1 -bs 1 -d fp16 -kvd fp16 -m llama-2-7b -mp /data/Llama-2-7B-Chat-hf-xft/ -tp /data/Llama-2-7B-Chat-hf/ -in 1024 -out 512 -i 1
```
INT8精度
```
bash run_benchmark.sh -s 1 -bs 1 -d w8a8 -kvd int8 -m llama-2-7b -mp /data/Llama-2-7B-Chat-hf-xft/ -tp /data/Llama-2-7B-Chat-hf/ -in 1024 -out 512 -i 1
```

Xeon SPR HBM的 AMX支持BF16和INT8，不支持FP16。从Xeon 6 开始，AMX支持FP16。

在SNC disable（QUAD mode）情况下，1 socket 9460跑Llama2-7B 

FP16是27 token/s，376 GB/s HBM带宽。

BF16是25 token/s， 349 GB/s HBM带宽。

INT8是43.7token/s，310.4 GB/s HBM带宽。

在SNC disable（QUAD mode）情况下，1 socket 9460跑Llama2-13B 

FP16是15.9 token/s，427.5 GB/s HBM带宽。

BF16是14.41 token/s， 388 GB/s HBM带宽。

INT8是26.07 token/s，355 GB/s HBM带宽。

在SNC-4情况下，1 socket 9460跑Llama2-13B 

FP16是18.74 token/s，504.2 GB/s HBM带宽。

BF16是16.51 token/s， 450.1 GB/s HBM带宽。

INT8是29.44 token/s，401 GB/s HBM带宽。

在SNC-4情况下，1 socket 9460跑Llama2-13B 用DDR 4800MT/s，理论最高带宽4.8*8*8=307.2GB/s。

FP16是10 token/s，260 GB/s DDR带宽。

BF16是9.69 token/s，252 GB/s DDR带宽。

（1）对比HBM和DDR，带宽限制了LLM推理性能。（2）不用AMX，用FP16反而HBM带宽利用率更高。这是为什么呢？AMX送数据有延迟，还是AMX使得CPU功耗高，导致HBM分到的功耗少，降低了HBM带宽？

BF16的处理效率没有FP16高，如果相同的带宽下，BF16 18.49 跟FP16 18.74 接近了，瓶颈在带宽。

基于QUAD mode (SNC disable)的，7B模型偏小并不能完整利用HBM的内存带宽，13B会有更好的结果。HBM SNC4+ 13B带宽可以达到450+ GB/s Per socket。

