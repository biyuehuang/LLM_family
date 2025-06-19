这是双卡BMG的使用指南

## 第一步安装ISO

Arc A770可以安装打包好的ISO，一步到位：http://joez.intel.com:5909/iso/multi-arc

(option)如果想手动安装相关库，Ubuntu24.10可以参考GenerateISO_Image_for_BMG.txt

(推荐)如果想一键安装相关库，Ubuntu25.04 可以直接运行 $sudo ./install-bmg-env.sh

 Ubuntu 25.04 desktop ISO:  https://releases.ubuntu.com/25.04/ubuntu-25.04-desktop-amd64.iso

## 第二步联网安装docker

https://docs.docker.com/engine/install/ubuntu/

===========
如果报错docker: Error response from daemon: Get "https://registry-1.docker.io/v2/": net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)

解决方法
```
sudo nano /etc/systemd/system/docker.service.d/http-proxy.conf

[Service]
Environment="HTTP_PROXY=http://proxy.example.com:8080" "HTTPS_PROXY=http://proxy.example.com:8080"

sudo systemctl daemon-reload
sudo systemctl restart docker
```
=============

## 第三步下载模型
modelscope或者huggingface下载32B模型，把它放在 /home/models
                                                                                                                                         
## 第四步创建容器
下载本目录的脚本run.sh，注意修改镜像名称，模型路径，容器名称。
2025/06/09 更新镜像：0.2.0-b4 (for BMG platform)
```                                                                                                                                         
sudo docker pull intelanalytics/ipex-llm-serving-xpu:0.2.0-b4

./run.sh                                                                                                                                         
                                                                                                                                         
sudo docker start bmg-test

sudo docker exec -it bmg-test bash    
```
创建容器后，默认在 /llm 路径下。

start-vllm-service.sh修改模型路径，如果是双卡把TENSOR_PARALLEL_SIZE改成2。
```
vim start-vllm-service.sh

MODEL_PATH=${MODEL_PATH:-"/llm/models/DeepSeek-R1-Distill-Qwen-32B"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"DeepSeek-R1-Distill-Qwen-32B"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
```
## 第五步运行benchmark测试

第一个terminal窗口运行服务
```
bash start-vllm-service.sh    

INFO:     Started server process [2044]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```
第二个terminal窗口运行benchmark测试，注意模型名称，4是并发路数。
```
python vllm_online_benchmark.py DeepSeek-R1-Distill-Qwen-32B 4   
```                                                                                                                                         
或者服务后，运行benchmark_serving的脚本

```
./run_benchmark_serving.sh
```

## 附录用xpu-smi查看独显的状态                                                                                                                                         
```                                                                                                                                         
apt install -y intel-gsc libigsc0 libmetee4 libpciaccess0

dpkg -i xpu-smi_3.0.35_20250404.151421.18c261db~u22.04_amd64.deb

watch -n 1 xpu-smi stats -d 0

watch -n 1 xpu-smi stats -d 1   
```                                                                                                                                         
