run ceval


```
conda create -n ipex-llm python=3.11 libuv
conda activate ipex-llm
pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/xpu -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tiktoken blobfile transformers==4.45 trl==0.11 accelerate==0.26.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --pre --upgrade ipex-llm bigdl-core-xe-all

set SYCL_CACHE_PERSISTENT=1 
set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 

python eval.py --model_path C:/Users/Lengda/Documents/DeepSeek-R1-Distill-Qwen-7B --eval_type validation --device xpu --eval_data_path ./data --qtype sym_int4
```
