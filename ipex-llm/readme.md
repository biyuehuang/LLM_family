安装环境：

https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Overview/install.md

1 运行在Intel iGPU Windows 11, 比如MTL，ADL，RPL
```
cd C:\Program Files (x86)\Intel\oneAPI
setvars.bat

set BIGDL_LLM_XMX_DISABLED=1
set SYCL_CACHE_PERSISTENT=1
python example3.py
```
