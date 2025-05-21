这是双卡A770的使用指南

更多更新参考：https://github.com/intel-analytics/ipex-llm/tree/main/docker/llm/serving/xpu/docker#vllm-serving-engine

推荐Ubuntu22.04，kernel 6.8/6.5，需要更新out-of-tree: https://dgpu-docs.intel.com/driver/client/overview.html

20250214 updated: intelanalytics/ipex-llm-serving-xpu:2.2.0-b13包含vLLM 0.6.6，Pytorch 2.6，OneAPI2025.0，oneCCL patch 0.0.6.6.

## (1) Run vllm offline TP continue batching

Download dataset from https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json

```
docker pull intelanalytics/ipex-llm-serving-xpu:2.2.0-b13

docker run -itd --net=host --device=/dev/dri -v /home/name/models:/llm/models -e no_proxy=localhost,127.0.0.1 --name=arc_vllm_server --shm-size="16g" intelanalytics/ipex-llm-serving-xpu:2.1.0b2

apt-get install jq

docker start arc_vllm_server

docker exec -it arc_vllm_server bash

nohup ./benchmark_throughput.sh > arc_2card_llama2-13b-int4-ccl.log 2>&1 &
```
查看XPU状态
```
sudo xpu-smi dump -m 1,2,18,22,26,31,34
```

## （2）Run vllm servering online TP continue batching



```
docker pull intelanalytics/ipex-llm-serving-xpu:2.2.0-b13

docker run -itd --net=host --device=/dev/dri -v /home/models:/llm/models -e no_proxy=localhost,127.0.0.1 --name=vllm_server_arc --shm-size="16g" intelanalytics/ipex-llm-serving-xpu-vllm-0.5.4-experimental:2.2.0b1

docker start vllm_server_arc

docker exec -it vllm_server_arc bash

apt-get install jq
```

打开一个terminal

```
bash start_Meta-Llama-3-8B-Instruct_serving.sh
```

打开另一个terminal

```
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "Meta-Llama-3-8B-Instruct",
"prompt": "San Francisco is a",
"max_tokens": 128,
"temperature": 0
}' | jq '.choices[0].text'
```

打开另一个terminal：
```
python vllm_online_benchmark.py Meta-Llama-3-8B-Instruct 16

Average first token latency: 2300.7708935313076 milliseconds.
P90 first token latency: 3740.910985499977 milliseconds.
P95 first token latency: 3742.0932860004996 milliseconds.

Average next token latency: 60.306824265625 milliseconds.
P90 next token latency: 62.515494169413195 milliseconds.
P95 next token latency: 63.976247517941594 milliseconds.
```
查看当前默认的环境路径
```
$ docker inspect vllm_server_arc
```

关掉端口8000
```
kill -9 $(lsof -t -i:8000)
```
