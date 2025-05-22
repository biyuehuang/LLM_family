#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import time
from transformers.utils import logging
from transformers import AutoTokenizer
import uvicorn
import asyncio
import argparse
from fastapi_server import FastApp
from fastapi_server import ModelWorker

logger = logging.get_logger(__name__)


async def main():
    parser = argparse.ArgumentParser(description='Predict Tokens using fastapi by leveraging ipex-llm')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf` and `meta-llama/Llama-2-70b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--port', type=int, default=8000,
                        help='The port number on which the server will run.')
    parser.add_argument('--device', type=str, default="GPU",
                        help='The device on which the model will run.')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    checkpoint = os.path.join(model_path, "DeepSeek-R1-Distill-Qwen-32B-ov")
    assistant_checkpoint = os.path.join(model_path, "DeepSeek-R1-Distill-Qwen-1.5B-ov")
    local_model = ModelWorker(checkpoint, assistant_checkpoint, device=args.device)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
   # prompt = "<｜User｜>Write a quicksort algorithm in Python. Write code only.<｜Assistant｜><think>"
    prompt = "hi"
    local_model.warmup(prompt)
    myapp = FastApp(local_model, tokenizer)
    config = uvicorn.Config(app=myapp.app, host="0.0.0.0", port=args.port)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
