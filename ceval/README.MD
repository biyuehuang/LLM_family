run ceval

Step 1: download dataset
```
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
mkdir data
mv ceval-exam.zip data
cd data; unzip ceval-exam.zip
```

Step 2: download model
```
pip install modelscope

modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --local_dir ./DeepSeek-R1-Distill-Qwen-14B
```

Step 3: run validation

framework 1: ipex-llm
```
conda create -n ipex-llm python=3.11 libuv
conda activate ipex-llm
pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/xpu -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tiktoken blobfile transformers==4.45 trl==0.11 accelerate==0.26.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --pre --upgrade ipex-llm bigdl-core-xe-all
pip install thefuzz

set SYCL_CACHE_PERSISTENT=1 
set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 

python eval.py --model_path C:/Users/Lengda/Documents/DeepSeek-R1-Distill-Qwen-14B --eval_type validation --device xpu --eval_data_path ./data --qtype sym_int4

python eval.py --model_path C:/Users/Lengda/Documents/DeepSeek-R1-Distill-Qwen-7B --eval_type validation --device xpu --eval_data_path ./data --qtype sym_int4
```

framework 2: openvino
```
conda create -n ov python=3.11
conda activate ov
pip install -r requirements_ov.txt
pip uninstall openvino openvino-tokenizers openvino-genai
pip install openvino==2025.1.0dev20250228 openvino-tokenizers==2025.1.0dev20250228 openvino-genai==2025.1.0dev20250228 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
pip install thefuzz
python eval.py --model_path C:/Users/Lengda/Documents/DeepSeek-R1-Distill-Qwen-14B --eval_type validation --device GPU --eval_data_path ./data --qtype sym_int4 --fw ov
```

framework 3: ollama
```
conda create -n ollama python=3.11
conda activate ollama
pip install thefuzz
python eval_ollama.py --model_path C:/Users/Lengda/Documents/DeepSeek-R1-Distill-Qwen-14B --eval_type validation --device GPU --eval_data_path ./data --qtype q4km
```



