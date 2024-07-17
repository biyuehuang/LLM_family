在windows 11 Pro平台验证过，GPU driver 5762。
如果是Linux，请参考：https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Quickstart/ollama_quickstart.md

下载并安装oneAPI 2024.2

```
conda create -n llm-cpp python=3.11
conda activate llm-cpp
pip install --pre --upgrade ipex-llm[cpp]
mkdir ollama
cd ollama
```

## 用管理员权限打开一个miniforge prompt，保持serve运行中。OLLAMA_NUM_GPU=999是使用GPU, OLLAMA_NUM_GPU=0是使用CPU
```
conda activate llm-cpp
cd C:\Program Files (x86)\Intel\oneAPI
setvars.bat
cd ollama
init-ollama.bat

set OLLAMA_NUM_GPU=999
set no_proxy=localhost,127.0.0.1
set ZES_ENABLE_SYSMAN=1
set SYCL_CACHE_PERSISTENT=1
ollama serve
```

## 第一种使用CURL，自动下载模型到C:\Users\u\.ollama\models。
### 管理员权限打开另一个miniforge prompt
```
conda activate llm-cpp
cd C:\Program Files (x86)\Intel\oneAPI
setvars.bat
cd ollama
ollama.exe pull qwen2:1.5b

curl http://localhost:11434/api/generate -d "{\"model\": \"qwen2:1.5b\", \"prompt\": \"Why is the sky blue?\", \"stream\": false,\"options\": {\"num_predict\": 100}} "
```

## 第二种使用GGUF，手动到huggingface下载模型。
### 管理员权限打开另一个miniforge prompt
```
conda activate llm-cpp
cd C:\Program Files (x86)\Intel\oneAPI
setvars.bat
cd ollama

set no_proxy=localhost,127.0.0.1
ollama create example -f Modelfile

ollama run example
```
