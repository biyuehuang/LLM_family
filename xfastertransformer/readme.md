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
docker exec -it “docker id” bash
docker start 14
```
模型转换指令
```
python -c 'import xfastertransformer as xft; xft.LlamaConvert().convert("${HF_DATASET_DIR}","${OUTPUT_DIR}")'
```

运行模型推理
```
cd /root/xFasterTransformer/benchmark
bash run_benchmark.sh -s 1 -bs 1 -d bf16 -kvd fp16 -m llama-2-7b -mp /data/Llama-2-7B-Chat-hf-xft/ -tp /data/Llama-2-7B-Chat-hf/ -in 1024 -out 512 -i 1
```
