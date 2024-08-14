(1)Run vllm offline TP continue batching
Download dataset from https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json

```
docker pull intelanalytics/ipex-llm-serving-xpu:2.1.0b2

docker run -itd --net=host --device=/dev/dri -v /opt:/opt -e no_proxy=localhost,127.0.0.1 --name=arc_vllm_server --shm-size="16g" intelanalytics/ipex-llm-serving-xpu:2.1.0b2

apt-get install jq

docker start arc_vllm_server

docker exec -it arc_vllm_server bash

nohup ./benchmark_throughput.sh > arc_2card_llama2-13b-int4-ccl.log 2>&1 &
```
查看XPU状态
```
sudo xpu-smi dump -m 1,2,18,22,26,31,34
```
