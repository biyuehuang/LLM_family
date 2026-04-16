1 Download llama-cpp from github

2 解压后，

## 测试一 llama-bench
```
llama-bench.exe -m model.gguf -ngl 999 -t 16 -p 512 -n 128 --mmap 0
```
## 测试二 llama-server多路并发
```
#### up to 4 concurrent requests, each with 4096 max context
llama-server -m "C:\Users\user\Desktop\model\gpt-oss-20b-Q4_K_M.gguf" -e -s 0 --host 127.0.0.1 --port 8080 -n 50 -mg 0 -ngl 999 --no-context-shift --slot-save-path "C:\Users\user\slot" -fa on --swa-full -c 16384 --ignore-eos -np 4 

or 

llama-server -m model.gguf -ngl 999 -c 16384 -np 4 
```
另一个窗口运行
```
pip install requests

python parallel-server-request.py 4 1k.txt
```
4代表并发数
