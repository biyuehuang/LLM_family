Run Deepseek-R1-32B by openvino

## Environment Preparation
```
conda create -n ov python=3.10
conda activate ov

# Install OpenVINO
pip install --upgrade --pre openvino openvino-tokenizers openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

pip install optimum-intel[nncf]

pip install fastapi uvicorn openai
```

## Models
- `DeepSeek-R1-Distill-Qwen-32B` is around 60G, loading the original model for run-time quantization may out of memory on ARL-H 64G machine. So it is recommended to save the int4 model beforehand.

```
optimum-cli export openvino --model C:/DeepSeek-R1-Distill-Qwen-32B --weight-format int4 --task text-generation-with-past DeepSeek-R1-Distill-Qwen-32B-ov

```

- `lightweight_serving.py` launches the service for the speculative decoding application and provides OpenAI-compatible endpoints (i.e. `/v1/chat/completions` and `/v1/models`).


Launch lightweight serving backend, repo-id-or-model-path is the parent directory of the two models

Default address is http://ip:8000/v1

```
cd open-webui
python lightweight_serving.py --repo-id-or-model-path C:/Users/models
```

## If use open-webui as chat frontend
```
conda create -n open-webui python=3.10

conda activate open-webui 
pip install open-webui=0.5.20

set http_proxy=
set https_proxy=
open-webui serve
```

If use openwebui, add the OpenAI API http://127.0.0.1:8000/v1 in [Admin Panel->Settings->Connection].

## If use gradio as chat frontend
```
pip install gradio

python gradio_webui.py --model-url http://127.0.0.1:8000/v1
```


