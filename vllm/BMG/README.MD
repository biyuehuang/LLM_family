这是双卡BMG的使用指南

双卡要注意设置 Bios setting: PCIe X16 Lane configuration > Slot1 8x +Slot 2 8x

## 第一步安装ISO

Arc A770可以安装打包好的ISO，一步到位：http://joez.intel.com:5909/iso/multi-arc

(option)如果想手动安装相关库，Ubuntu24.10可以参考GenerateISO_Image_for_BMG.txt

(推荐)如果想一键安装相关库，Ubuntu25.04 可以直接运行本目录的脚本 $sudo ./install-bmg-env.sh

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

2025/07/07 更新镜像：0.2.0-b6 (for BMG platform)

2025/06/09 更新镜像：0.2.0-b4 (for BMG platform)
```                                                                                                                                         
sudo docker pull intelanalytics/ipex-llm-serving-xpu:0.2.0-b6

sudo ./run.sh                                                                                                                                         
                                                                                                                                         
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

#### 第一个terminal窗口运行服务
```
bash start-vllm-service.sh    
```
等待几分钟，输出信息如下，说明服务已启动成功：
```
INFO:     Started server process [2044]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```
#### 第二个terminal窗口运行benchmark测试，注意模型名称，4是并发路数。
```
python vllm_online_benchmark.py DeepSeek-R1-Distill-Qwen-32B 4   
```   
输出信息如下：
```
running bench.py
model_name: DeepSeek-R1-Distill-Qwen-32B
max_seq: 4
input_length: 1024
output_length: 512
[W520 14:52:24.692600723 OperatorEntry.cpp:154] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_validate_compressed_sparse_indices(bool is_crow, Tensor compressed_idx, Tensor plain_idx, int cdim, int dim, int nnz) -> ()
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /pytorch/build/aten/src/ATen/RegisterCPU.cpp:30477
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:468 (function operator())
[W520 14:52:26.072611582 OperatorEntry.cpp:154] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_validate_compressed_sparse_indices(bool is_crow, Tensor compressed_idx, Tensor plain_idx, int cdim, int dim, int nnz) -> ()
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /pytorch/build/aten/src/ATen/RegisterCPU.cpp:30477
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:468 (function operator())
INFO 05-20 14:52:26 [__init__.py:239] Automatically detected platform xpu.
Token indices sequence length is longer than the specified maximum sequence length for this model (34276 > 16384). Running this sequence through the model will result in indexing errors
Warm Up: 100%|███████████████████████████████████████████████████████| 2/2 [00:06<00:00,  3.41s/req]
Warm Up: 100%|███████████████████████████████████████████████████████| 8/8 [01:25<00:00, 10.72s/req]
Benchmarking: 100%|████████████████████████████████████████████████| 16/16 [02:50<00:00, 10.69s/req]
Total time for 16 requests with 4 concurrent requests: 170.9960746229999 seconds.
Average responce time: 42.748967152562486
Token throughput: 47.907532486117844

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py:3504: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
/usr/local/lib/python3.11/dist-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
Average first token latency: 3063.9210203124776 milliseconds.
P90 first token latency: 4468.496373999983 milliseconds.
P95 first token latency: 4470.045157500067 milliseconds.

Average next token latency: 77.66146085163899 milliseconds.
P90 next token latency: 81.38692605283755 milliseconds.
P95 next token latency: 81.40906725293529 milliseconds.
```

#### 或者运行服务后，运行benchmark_serving的脚本。本目录的脚本run_benchmark_serving.sh

```
./run_benchmark_serving.sh
```

## 附录用xpu-smi查看独显的状态                                                                                                                                         
```                                                                                                                                         
sudo apt install -y intel-gsc libigsc0 libmetee4 libpciaccess0

sudo dpkg -i xpu-smi_3.0.35_20250404.151421.18c261db~u22.04_amd64.deb

sudo watch -n 1 xpu-smi stats -d 0

sudo watch -n 1 xpu-smi stats -d 1   
```                                                                                                                                         
