https://github.com/openvinotoolkit/openvino.genai/tree/master

```
conda create -n ov-llm-bench-env python=3.11
conda activate ov-llm-bench-env
pip install --upgrade pip

git clone  https://github.com/openvinotoolkit/openvino.genai.git
cd openvino.genai/tools/llm_bench
pip install -r requirements.txt  
```
option
```
pip uninstall -y openvino openvino-genai openvino_tokenizers
pip install --pre openvino==2025.1.0rc2 openvino-tokenizers==2025.1.0.0rc2 openvino-genai==2025.1.0.0rc2 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release
```

convert model

INT4
```
