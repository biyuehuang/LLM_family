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

### 报错记录和解决方法：
(1) import torch出现undefined symbol: iJIT_NotifyEvent
```
File "/m/venv/lib/python3.10/site-packages/torch/init.py", line 229, in
from torch._C import * # noqa: F403
ImportError: /m/venv/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```
解决方法：首先检查是不是mkl版本导致的，
```
pip install mkl==2024.0
```
如果改变mkl版本不能解决，那么这个报错大概率来自oneapi和ipex版本不匹配。如果已经安装了oneapi 2024.2，可以升级ipex：
```
python -m pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30.post0 oneccl_bind_pt==2.1.300+xpu --extra-index-url 
https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```
