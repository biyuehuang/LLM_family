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
from transformers.utils import logging
import time
import asyncio
#from transformers import TextIteratorStreamer
import openvino_genai
import queue
import threading
from typing import Union, List
logger = logging.get_logger(__name__)



class IterableStreamer(openvino_genai.StreamerBase):
    """
    A custom streamer class for handling token streaming and detokenization with buffering.

    Attributes:
        tokenizer (Tokenizer): The tokenizer used for encoding and decoding tokens.
        tokens_cache (list): A buffer to accumulate tokens for detokenization.
        text_queue (Queue): A synchronized queue for storing decoded text chunks.
        print_len (int): The length of the printed text to manage incremental decoding.
    """

    def __init__(self, tokenizer):
        """
        Initializes the IterableStreamer with the given tokenizer.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for encoding and decoding tokens.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_cache = []
        self.text_queue = queue.Queue()
        self.print_len = 0
        self.decoded_lengths = []

    def __iter__(self):
        """
        Returns the iterator object itself.
        """
        return self

    def __next__(self):
        """
        Returns the next value from the text queue.

        Returns:
            str: The next decoded text chunk.

        Raises:
            StopIteration: If there are no more elements in the queue.
        """
        # get() will be blocked until a token is available.
        value = self.text_queue.get()
        if value is None:
            raise StopIteration
        return value

    def get_stop_flag(self):
        """
        Checks whether the generation process should be stopped or cancelled.

        Returns:
            openvino_genai.StreamingStatus: Always returns RUNNING in this implementation.
        """
        return openvino_genai.StreamingStatus.RUNNING

    def write_word(self, word: str):
        """
        Puts a word into the text queue.

        Args:
            word (str): The word to put into the queue.
        """
        self.text_queue.put(word)

    def write(self, token: Union[int, List[int]]) -> openvino_genai.StreamingStatus:
        """
        Processes a token and manages the decoding buffer. Adds decoded text to the queue.

        Args:
            token (Union[int, List[int]]): The token(s) to process.

        Returns:
            bool: True if generation should be stopped, False otherwise.
        """
        if type(token) is list:
            self.tokens_cache += token
            self.decoded_lengths += [-2 for _ in range(len(token) - 1)]
        else:
            self.tokens_cache.append(token)

        text = self.tokenizer.decode(self.tokens_cache)
        self.decoded_lengths.append(len(text))

        word = ""
        delay_n_tokens = 3
        if len(text) > self.print_len and "\n" == text[-1]:
            # Flush the cache after the new line symbol.
            word = text[self.print_len :]
            self.tokens_cache = []
            self.decoded_lengths = []
            self.print_len = 0
        elif len(text) > 0 and text[-1] == chr(65533):
            # Don't print incomplete text.
            self.decoded_lengths[-1] = -1
        elif len(self.tokens_cache) >= delay_n_tokens:
            self.compute_decoded_length_for_position(
                len(self.decoded_lengths) - delay_n_tokens
            )
            print_until = self.decoded_lengths[-delay_n_tokens]
            if print_until != -1 and print_until > self.print_len:
                # It is possible to have a shorter text after adding new token.
                # Print to output only if text length is increased and text is complete (print_until != -1).
                word = text[self.print_len : print_until]
                self.print_len = print_until
        self.write_word(word)

        stop_flag = self.get_stop_flag()
        if stop_flag != openvino_genai.StreamingStatus.RUNNING:
            # When generation is stopped from streamer then end is not called, need to call it here manually.
            self.end()

        return stop_flag

    def compute_decoded_length_for_position(self, cache_position: int):
        # decode was performed for this position, skippping
        if self.decoded_lengths[cache_position] != -2:
            return

        cache_for_position = self.tokens_cache[: cache_position + 1]
        text_for_position = self.tokenizer.decode(cache_for_position)

        if len(text_for_position) > 0 and text_for_position[-1] == chr(65533):
            # Mark text as incomplete
            self.decoded_lengths[cache_position] = -1
        else:
            self.decoded_lengths[cache_position] = len(text_for_position)

    def end(self):
        """
        Flushes residual tokens from the buffer and puts a None value in the queue to signal the end.
        """
        text = self.tokenizer.decode(self.tokens_cache)
        if len(text) > self.print_len:
            word = text[self.print_len :]
            self.write_word(word)
            self.tokens_cache = []
            self.print_len = 0
        self.text_queue.put(None)


class ChunkStreamer(IterableStreamer):

    def __init__(self, tokenizer, tokens_len):
        super().__init__(tokenizer)
        self.tokens_len = tokens_len

    def write(self, token: Union[int, List[int]]) -> openvino_genai.StreamingStatus:
        if (len(self.tokens_cache) + 1) % self.tokens_len == 0:
            return super().write(token)

        if type(token) is list:
            self.tokens_cache += token
            # -2 means no decode was done for this token position
            self.decoded_lengths += [-2 for _ in range(len(token))]
        else:
            self.tokens_cache.append(token)
            self.decoded_lengths.append(-2)

        return openvino_genai.StreamingStatus.RUNNING

class ModelWorker:
    def __init__(self, checkpoint, device="GPU"):
        self.device = device
        start = time.perf_counter()

        scheduler_config = openvino_genai.SchedulerConfig()
        scheduler_config.cache_size = 1
        scheduler_config.enable_prefix_caching = False
        scheduler_config.max_num_batched_tokens = 2147483647
        self.model  = openvino_genai.LLMPipeline(checkpoint, device, {"scheduler_config": scheduler_config})

        self.config = openvino_genai.GenerationConfig()
        self.config.max_new_tokens = 2048


        chat_template = \
            "{% for message in messages %}"\
            "{% if (message['role'] == 'user') %}"\
            "{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n<|im_start|>assistant\n'}}"\
            "{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|im_end|>\n'}}"\
            "{% endif %}"\
            "{% endfor %}"
        self.model.get_tokenizer().set_chat_template(chat_template)

        end = time.perf_counter()
        logger.info(f"Time to load weights: {end - start:.2f}s")
        self.waiting_requests = asyncio.Queue()
        self.streamer = {}
        model_name = os.path.basename(checkpoint).replace("-ov", "")
        self.model_name = model_name

    def warmup(self, prompt):
        for i in range(1):
            st = time.time()
            #self.model.generate("hi", self.config, self.streamer)
            result = self.model.generate(prompt, self.config)
            print(result)
            et = time.time()
            print(et - st)

  
    async def add_request(self):
        if self.waiting_requests.empty():
            return
        tmp_result = await self.waiting_requests.get()
        request_id, prompt_request = tmp_result
        plain_texts = prompt_request.inputs
        #print("Prompt: ", plain_texts)
        #inputs_embeds = None
       # inputs = tokenizer(plain_texts, return_tensors="pt")
       # input_ids = inputs.input_ids
        parameters = prompt_request.parameters
        return plain_texts, parameters, request_id


    async def process_step(self, result_dict, processor=None):
        if not self.waiting_requests.empty():
            plain_texts, parameters, request_id = \
                await self.add_request()
            tokens_len = 10  # chunk size
            self.streamer[request_id] = ChunkStreamer(self.model.get_tokenizer(), tokens_len)
            def model_generate():
                self.model.generate(plain_texts, self.config, self.streamer[request_id])
            
            t1 = threading.Thread(target=model_generate)
            t1.start()
