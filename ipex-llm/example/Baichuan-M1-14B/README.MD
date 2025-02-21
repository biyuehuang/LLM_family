同名文件

【ipex-2.6】
```
conda create -n env-2.6 python=3.10 libuv ffmpeg

pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/xpu

cd AIGC\resources

pip install -r requirement-ipex-2.6.txt


pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/xpu

#pip install --pre --upgrade ipex-llm==2.2.0b20250211

pip install transformers==4.45 trl==0.11 accelerate==0.26.0 

pip install numpy==1.23 frontend tools


set SYCL_CACHE_PERSISTENT=1 
set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 
```
