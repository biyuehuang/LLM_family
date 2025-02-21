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
#  0124 replace ASR
#  0301 (1)lang ja. (2)tts 200 to 300. (3)support history 30G . (4)LLM baichaun2-7b chatglm3-6b en input.
#  0304 add 、 in def chat

# 音频文件需要时16k采样，转换为16k方法： ffmpeg -i input.wav -ar 16000 output.wav
# ffmpeg 切分16k音频方法： ffmpeg -i ~/whisper/chinese_5min3s.wav -ar 16000 -ss 00:00:00 -t 10s chinese_10s_16k.wav

# gradio UI loading 太慢 https://github.com/gradio-app/gradio/issues/4332#issuecomment-1563758104
# vim ~/miniconda3/envs/llm/lib/python3.9/site-packages/gradio/themes/utils/fonts.py # 注意需要修改一下虚拟环境的名字
# 大约在第50行的位置，注释return 那一行，并且在前面加pass
# def stylesheet(self) -> str:
#     pass
#     #return f'https://fonts.googleapis.com/css2?family={self.name.replace(" ", "+")}:wght@{";".join(str(weight) for weight in self.weights)}&display=swap'

import json
import os
from typing import Dict, List
import torch
import time
import sys
import traceback
import global_setup
from ipex_llm.transformers import AutoModelForCausalLM, AutoModel
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    STOPPING_CRITERIA_INPUTS_DOCSTRING,
    add_start_docstrings,
)
from transformers import (
    AutoTokenizer,
    TextIteratorStreamer,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)

import gc
import re
from paddlespeech.cli.tts.infer import TTSExecutor
import shutil
from multiprocessing import Queue
from datetime import datetime
import langid
import psutil
import customException
import threading  ## 0314 end
import rag
from sensitive_words_blocking import SensitiveWordsBlocking

os.environ["USE_XETLA"] = "OFF"
os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
os.environ["ENABLE_SDP_FUSION"] = "1"


chat_templetes = {
    "Baichuan-M1-14B-Instruct": {
        "en": "Answer my questions in English.\n\nQuestion: {prompt}\n",
        "zh": "请使用中文回答我的问题。\n\n问: {prompt}\n",
        "rag_en": "Important: Answer questions in English. Please answer my question according to the reference information below. Reference information is as follows:\n{context}\n\nQuestion: {prompt}\n",
        "rag_zh": "重要：使用中文回答问题。请根据一下参考信息来回答我的问题。参考信息如下：\n{context}\n\n问: {prompt}\n",
    },
    "Qwen2-7B-Instruct": {
        "en": "Answer my questions in English.\n\nQuestion: {prompt}\n",
        "zh": "根据用户提问回答问题。回答的语种与问题的语种一致。\n\n问: {prompt}\n",
        "rag_en": "Important: Answer questions in English. Please answer my question according to the reference information below. Reference information is as follows:\n{context}\n\nQuestion: {prompt}\n",
        "rag_zh": "重要：使用中文回答问题。请根据一下参考信息来回答我的问题。参考信息如下：\n{context}\n\n问: {prompt}\n",
    },
}


chat_model: PreTrainedModel
# check if cl_cache exist or not
cache_dir = os.path.expanduser("~/cl_cache")
tts_executor: TTSExecutor | None = None
chat_thread: threading.Thread = None
audio_thread: threading.Thread = None
generating: bool = False
sensitive_words_filter: SensitiveWordsBlocking = SensitiveWordsBlocking()

## 0314 end

if not os.path.exists(cache_dir):
    # create the folder if it does not exist
    os.makedirs(cache_dir)


def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)

    info = p.memory_full_info()
    memory = info.uss / 1024.0 / 1024
    print("******************* {} memory used: {} MB".format(hint, memory))
    chatglm2_model_path = model_path + "/" + model_name_llm + "-int4"
    print("loading LLM---------", chatglm2_model_path)
    t2 = time.time()

    if model_name_llm == "chatglm2-6b" or model_name_llm == "chatglm3-6b":
        chatglm2_model = AutoModel.load_low_bit(
            chatglm2_model_path,
            trust_remote_code=True,
            optimize_model=True,
            use_cache=True,
            replace_embedding=True,
        ).eval()
    else:
        chatglm2_model = AutoModelForCausalLM.load_low_bit(
            chatglm2_model_path,
            trust_remote_code=True,
            optimize_model=True,
            use_cache=True,
            replace_embedding=True,
        ).eval()

    # chatglm2_model =  AutoModel.load_low_bit(chatglm2_model_path, trust_remote_code=True, optimize_model=True,use_cache=True,replace_embedding=True).eval()
    # chatglm2_model = AutoModel.from_pretrained(chatglm2_model_path, trust_remote_code=True, optimize_model=True, load_in_4bit=True).eval()
    chatglm2_model.to(device)

    ## for MTL iGPU
    # if device == global_setup.DEVICE:
    #    chatglm2_model.transformer.embedding.to('cpu')

    tokenizer = AutoTokenizer.from_pretrained(
        chatglm2_model_path, trust_remote_code=True
    )
    torch.xpu.synchronize()
    t3 = time.time()
    print("loading LLM---------Done, cost time(s): ", chatglm2_model_path, t3 - t2)
    return chatglm2_model, tokenizer


# 使用TTS将文字转换为语音文件
def load_tts_model_paddle(text_in="今天的天气不错啊", audio_out="./output.wav"):
    home_path = os.environ["PPSPEECH_HOME"]
    voc_ckpt = os.path.abspath(
        os.path.join(
            home_path,
            "models/hifigan_csmsc_onnx-zh/1.0/hifigan_csmsc_onnx_0.2.0/hifigan_csmsc.onnx",
        )
    )
    am_ckpt = os.path.abspath(
        os.path.join(
            home_path,
            "models/fastspeech2_mix_onnx-mix/2.0/fastspeech2_mix_onnx_0.2.0/fastspeech2_mix.onnx",
        )
    )
    phones_dict = os.path.abspath(
        os.path.join(
            home_path,
            "models/fastspeech2_mix_onnx-mix/2.0/fastspeech2_mix_onnx_0.2.0/phone_id_map.txt",
        )
    )
    with torch.inference_mode():
        tts_executor(
            text=text_in,
            output=audio_out,
            am="fastspeech2_mix",
            am_ckpt=am_ckpt,
            voc="hifigan_csmsc",
            voc_ckpt=voc_ckpt,
            phones_dict=phones_dict,
            spk_id=174,
            lang="mix",
            use_onnx=True,
        )
    print("***********Wave file has been generated: {}".format(audio_out))


class StopOnStringCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings: List[str]):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # 将input_ids转换为字符串
        text = self.tokenizer.decode(
            input_ids[0], skip_prompt=True, skip_special_tokens=True
        )
        # 检查文本中是否包含停止字符串
        for stop_string in self.stop_strings:
            print(f"stop by string ：{text}")
            if stop_string in text:
                return True
        return False


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops: List[int]):
        self.stops = stops
        StoppingCriteria.__init__(self)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if stop in input_ids:
                return True
        return False


def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == "Qwen":
        stop_words_ids = [151645, 151643]
    elif chat_format == "ChatGLM3":
        stop_words_ids = [
            tokenizer.eos_token_id,
            tokenizer.get_command("<|user|>"),
            tokenizer.get_command("<|observation|>"),
            tokenizer.get_command("<|assistant|>"),
        ]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids


## 1220 end


def stream_chat_generate(model: PreTrainedModel, args: dict, error_callback=None):
    try:
        model.generate(**args)
    except Exception as ex:
        traceback.print_exc()
        if error_callback is not None:
            error_callback(ex)


def user_stop(input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
    global user_stop_flag
    return user_stop_flag


def stream_chat(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_tokens: int,
    device="xpu",
    error_callback=None,
):
    global chat_thread, user_stop_flag, model_name_llm, enable_memory_32g_feature

    input_ids = tokenizer([prompt], return_tensors="pt").to(device)
    print("stream_chat-----input_ids:", input_ids)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,  # skip prompt in the generated tokens
        skip_special_tokens=True,
    )

    stopping_criteria = StoppingCriteriaList()  ## 1220
    # if model_name_llm == "Qwen2-7B-Instruct":
    #     stopping_criteria.append(
    #         StoppingCriteriaSub(stops=get_stop_words_ids("Qwen", tokenizer=tokenizer))
    #     )
    # else:
    #     stopping_criteria.append(
    #         StopOnStringCriteria(tokenizer, ["<|im_start|>", "<|im_end|>"])
    #     )
    stopping_criteria.append(CustomStopCriteria(user_stop))

    generate_kwargs = dict(
        input_ids,
        streamer=streamer,
        num_beams=1,
        do_sample=False,
        max_new_tokens=max_tokens,
        stopping_criteria=stopping_criteria,
    )
    if model_name_llm == "Qwen2-7B-Instruct":
        print("-----Qwen2-7B")
        generate_kwargs = dict(
            input_ids,
            streamer=streamer,
            num_beams=1,
            do_sample=False,
            max_new_tokens=max_tokens,
            eos_token_id=[151645, 151643],
        )

    print("generate_kwargs:{}", generate_kwargs)

    chat_thread = threading.Thread(
        target=stream_chat_generate, args=(model, generate_kwargs, error_callback)
    )
    chat_thread.start()

    return streamer


def chat(
    prompt: str,
    histories: List[Dict[str, str]],
    langid: str,
    max_length: int,
    use_rag=False,
    rag_source=None,
    llm_first_token_callback=None,
    llm_after_token_callback=None,
    llm_model_name: str = None,
    text_out_callback=None,
    generate_audio=False,
    load_model_start_callback=None,
    load_model_finish_callback=None,
    error_callback=None,
):
    global \
        chat_model, \
        tokenizer, \
        model_path, \
        model_name_llm, \
        audio_generate_arg, \
        user_stop_flag, \
        enable_memory_32g_feature, \
        chat_templetes, \
        sensitive_words_filter

    rest_count = 0

    if llm_model_name != model_name_llm:
        if chat_model is not None:
            chat_model.to("cpu")
            torch.xpu.empty_cache()
            del chat_model
            gc.collect()

        model_name_llm = llm_model_name
        if model_name_llm == "Baichuan-M1-14B-Instruct":
            model_full_path = os.path.abspath(
                os.path.join(model_path, model_name_llm)
            )
            print(f"******* loading {model_name_llm} from {model_full_path}")
            if load_model_start_callback is not None:
                load_model_start_callback(model_name_llm)
            stm = time.time()    
            chat_model = AutoModelForCausalLM.from_pretrained(model_full_path, torch_dtype = torch.half, load_in_low_bit='sym_int4',trust_remote_code=True).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_full_path,trust_remote_code=True)    
            chat_model = chat_model.to('xpu')
        else:
            model_full_path = os.path.abspath(
                os.path.join(model_path, model_name_llm + "-int4")
            )
            print(f"******* loading {model_name_llm} from {model_full_path}")
            if load_model_start_callback is not None:
                load_model_start_callback(model_name_llm)
            stm = time.time()

            chat_model = AutoModelForCausalLM.load_low_bit(
                model_full_path,
                trust_remote_code=True,
                optimize_model=True,
                use_cache=True,
                replace_embedding=True,
            ).eval()

            tokenizer = AutoTokenizer.from_pretrained(
                model_full_path, trust_remote_code=True
            )

            chat_model.half().to(global_setup.device)  ## 0307

        print("********** model load time (s)= ", time.time() - stm)
        if load_model_finish_callback is not None:
            load_model_finish_callback(model_name_llm)

    torch.xpu.synchronize()
    timeStart = time.time()
    timeFirstRecord = False

    # rag query
    if enable_memory_32g_feature and use_rag and rag.Is_Inited:
        query_success, context, rag_source = rag.query(prompt)
    else:
        query_success = None
        rag_source = None

    templete = chat_templetes.get(llm_model_name)
    # print("-------------templete: ", templete, chat_templetes, llm_model_name)

    if query_success:
        question = templete.get(f"rag_{langid}").format_map(
            {"context": context, "prompt": prompt}
        )
    else:
        question = templete.get(langid).format_map({"prompt": prompt})

    histories.append({"role": "user", "content": question})
    # lastCheckPos = 0
    response = ""
    audio_response: str = ""
    is_baichuan_model = llm_model_name.startswith("Baichuan2-")

    generate_prompt: str = tokenizer.apply_chat_template(
        histories, tokenize=False, add_generation_prompt=True
    )

    if is_baichuan_model:
        generate_prompt = generate_prompt.replace("<|im_start|>", "<s>").replace(
            "<|im_end|>", "</s>"
        )

    history_max_token = 4096 if enable_memory_32g_feature else 2048

    while len(tokenizer.tokenize(generate_prompt)) > history_max_token:
        histories.remove(histories[0])
        generate_prompt = tokenizer.apply_chat_template(
            histories, tokenize=False, add_generation_prompt=True
        )

        if is_baichuan_model:
            generate_prompt = generate_prompt.replace("<|im_start|>", "<s>").replace(
                "<|im_end|>", "</s>"
            )

    print("prompt:\n", generate_prompt)

    print("response:")

    with torch.inference_mode():
        for stream_output in stream_chat(
            chat_model,
            tokenizer,
            generate_prompt,
            # prompt,
            max_length,
            global_setup.device,
            error_callback,
        ):
            # 检测用户是否主动退出
            if user_stop_flag:
                print("\n---------chat exit by user stop------")
                return
            stream_output = sensitive_words_filter.filter_all(stream_output)
            if text_out_callback is not None:
                text_out_callback(stream_output)

            print(stream_output, end="")
            response += stream_output
            audio_response += stream_output
            # Add a blinking cursor to simulate typing
            if not timeFirstRecord:
                torch.xpu.synchronize()
                timeFirst = time.time()  # - timeStart
                timeFirstRecord = True

            if generate_audio and len(tokenizer.tokenize(audio_response)) > 30:
                # 找到字符串中最后一个标点符号
                matches = list(re.finditer(r"[。！？：.!?:，,]", audio_response))
                sub_index = (
                    matches[-1].start() + 1 if matches else audio_response.__len__() - 1
                )
                segment = audio_response[:sub_index]
                audio_generate_arg.segment_queue.put(segment)
                audio_generate_arg.event.set()
                rest_count += 1
                audio_response = audio_response[sub_index:]
                print(f"音频队列入列{rest_count}")
            # token_num = len(tokenizer.tokenize(response))
            # if token_num > 300 and all_flag:  # 0301
            #     all_flag = False
            # elif generate_audio:
            #     # rest_count = count
            #     exists_punctuation = (
            #         re.match("[。！？：.!?:，,、]", response[rest_count:]) is not None
            #     )
            #     if exists_punctuation and token_num > 40 and all_flag:
            #         segment = re.split("[。！？：.!?:]", response[rest_count:])
            #         # print("1-------------segment: ", segment)
            #         if len(segment) > 1:
            #             audio_generate_arg.segment_queue.put(segment[-2])
            #             audio_generate_arg.event.set()
            #             rest_count += len(segment[-2]) + 1

            #     elif exists_punctuation and token_num <= 40 and all_flag:
            #         segment = re.split(
            #             "[。！？：.!?:，,、]", response[rest_count:]
            #         )  # 0304
            #         print("2-------------segment: ", segment, len(segment))
            #         if len(segment) > 1:
            #             audio_generate_arg.segment_queue.put(segment[-2])
            #             audio_generate_arg.event.set()
            #             rest_count += len(segment[-2]) + 1

        timeTotal = time.time()
        llm_time = timeTotal - timeStart
        token_count_input = len(tokenizer.tokenize(generate_prompt))
        token_count_output = len(tokenizer.tokenize(response))
        llm_ms_first_token = (timeFirst - timeStart) * 1000
        if token_count_output == 0 or token_count_output == 1:  ## 1220
            llm_ms_after_token = (timeTotal - timeFirst) * 1000  ### 1220
        elif token_count_output > 1:
            llm_ms_after_token = (
                (timeTotal - timeFirst) / (token_count_output - 1) * 1000
            )

        if generate_audio:
            audio_generate_arg.segment_queue.put(audio_response)
            audio_generate_arg.event.set()

    audio_generate_arg.out_finish = True

    if rag_source:
        if langid == "en":
            rag_source = f"\n\nsource file: {rag_source}"
        else:
            rag_source = f"\n\n来源文件: {rag_source}"

    if text_out_callback is not None and use_rag and rag_source is not None:
        text_out_callback(rag_source)

    print("\n")
    print("token count input: ", token_count_input)
    print("token count output: ", token_count_output)
    print("LLM First token latency(ms): ", llm_ms_first_token)
    print("LLM After token latency(ms/token): ", llm_ms_after_token)
    print("LLM time cost(s): ", llm_time)

    # llm_ms_first_token_list.append(llm_ms_first_token)
    # llm_ms_after_token_list.append(llm_ms_after_token)
    if llm_first_token_callback is not None:
        llm_first_token_callback(f"{round(llm_ms_first_token, 2)} ms")
    if llm_after_token_callback is not None:
        llm_after_token_callback(f"{round(llm_ms_after_token, 2)} ms/token")
    print("-" * 50)
    print("\n")


def get_speech(wav_path_callback):
    global audio_generate_arg, save_wav_root_path1
    audio_folder = os.path.join(os.getcwd(), save_wav_root_path1)
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)
    count = 0
    while True:
        while not user_stop_flag and not audio_generate_arg.segment_queue.empty():
            text: str = audio_generate_arg.segment_queue.get_nowait()
            if not user_stop_flag and text is not None and text != "":
                count = count + 1
                print(f"text in get_speech: {count}")
                audio_out = os.path.abspath(
                    os.path.join(
                        audio_folder,
                        "{}.mp3".format(
                            str(datetime.now()).replace(":", "-").replace(" ", "_")
                        ),
                    )
                )
                # asyncio.run(get_tts(text, audio_out))
                text = text.replace("ChatGLM2-6B", "chat G L M 2 - 6 B")
                text = text.replace("AI", "A.I")  ## 0131
                text = re.sub(r"<\|im_[^|]+\|>", "", text)
                for word in re.findall(
                    "\s[A-Z]+|[A-Z]+\s", text
                ):  # 往大写字母单词间加空格
                    new_word = re.sub(r"(?<=\w)(?=\w)", " ", word)
                    text = re.sub(word, new_word, text)

                load_tts_model_paddle(text, audio_out)

                if wav_path_callback is not None:
                    wav_path_callback(audio_out)

        # 当用户停止时，直接停止函数执行
        if user_stop_flag:
            print("get_speech exit by user stop")
            return
        elif not audio_generate_arg.out_finish:
            audio_generate_arg.event.clear()
            audio_generate_arg.event.wait()
        else:
            break

    print("get_speech normal exit")


def stop_generate():
    global chat_thread, audio_thread, audio_generate_arg, user_stop_flag, generating
    print("stop_generate start")
    # 设置全局用户停止标识，让各个线程快速退出
    user_stop_flag = True
    # 等待chat模型停止
    if chat_thread is not None and chat_thread.is_alive():
        print("waiting chat_thread stop")
        chat_thread.join()

    # 等待audio生成线程退出
    if audio_thread is not None and audio_thread.is_alive():
        print("waitting audio_thread stop")
        # 唤醒audio生成线程，让其能根据thread_cancel_flag的值，自动退出
        audio_generate_arg.event.set()
        audio_thread.join()
        audio_generate_arg.event.clear()

    # 清空队列
    while not audio_generate_arg.segment_queue.empty():
        audio_generate_arg.segment_queue.get()

    chat_thread = None
    audio_thread = None
    user_stop_flag = False
    generating = False
    print("stop_generate finish. will clear temp resource!")
    clear_audio_cache()


def clear_audio_cache():
    global save_wav_root_path1
    try:
        if os.path.exists(save_wav_root_path1):
            # 删除文件夹及其内容
            shutil.rmtree(save_wav_root_path1)
        # 新建文件夹
        os.makedirs(save_wav_root_path1)
        print(save_wav_root_path1, "****Folder deleted and recreated.")
    except Exception as e:
        print(f"An error occurred: {e}")


class thread_with_trace(threading.Thread):
    def __init__(self, *args, **keywords):
        threading.Thread.__init__(self, *args, **keywords)
        self.killed = False

    def start(self):
        self.__run_backup = self.run
        self.run = self.__run
        threading.Thread.start(self)

    def __run(self):
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def globaltrace(self, frame, event, arg):
        if event == "call":
            return self.localtrace
        else:
            return None

    def localtrace(self, frame, event, arg):
        if self.killed:
            if event == "line":
                raise SystemExit()
        return self.localtrace

    def kill(self):
        self.killed = True


def predict_user(
    prompt: str,
    histories: List[Dict[str, str]],
    input_lang=None,  # 0301
    text_out_callback=None,
    llm_model_name="chatglm2-6b",
    max_token=512,
    llm_first_token_callback=None,
    llm_after_token_callback=None,
    use_rag=False,
    generate_audio=True,
    audio_out_callback=None,
    load_model_start_callback=None,
    load_model_finish_callback=None,
    error_callback=None,
):  ##
    global \
        generating, \
        audio_thread, \
        audio_generate_arg, \
        chat_model, \
        tokenizer, \
        model_path, \
        model_name_llm, \
        sensitive_words_filter

    if generating:
        print("stop prev generate")
        # 停止上一次的生成
        stop_generate()

    generating = True

    try:
        print("-" * 50)
        print(
            f"user_input:{prompt} llm_model_name:{llm_model_name} max_token:{max_token} generate_audio:{generate_audio} use_rag:{use_rag}"
        )
        if input_lang is None:  # 0301
            input_lang = langid.classify(prompt)[0]  # 0301

        print("------language: ", input_lang)

        lang_prefix = input_lang[:2]
        # 未支持的语言处理
        if lang_prefix != "zh" and lang_prefix != "en":  ## 0131 0301
            print("This program only support Chinese and English!!!!!!!!!!!!!!!")
            if text_out_callback is not None:
                text_out_callback(
                    "无法识别内容。请使用中文或英文重新输入"
                    if lang_prefix != "zh"
                    else "only Chinese and English input is supported, please change the input language and try again"
                )

            if generate_audio and audio_out_callback is not None:
                audio_out_callback()

            if llm_first_token_callback is not None:
                llm_first_token_callback("")

            if llm_after_token_callback is not None:
                llm_after_token_callback("")

            return
        elif sensitive_words_filter.exists(prompt):
            if text_out_callback is not None:
                text_out_callback(
                    "语音内容包含违禁词汇"
                    if lang_prefix != "zh"
                    else "The voice content contains prohibited words"
                )
            raise customException.SensitiveWordsException()

        if prompt:
            # 需要生成语音
            if generate_audio:
                print("text with audio ouput")
                audio_generate_arg.out_finish = False
                audio_thread = thread_with_trace(
                    target=get_speech, args=[audio_out_callback]
                )
                audio_thread.start()
                chat(
                    prompt=prompt,
                    histories=histories,
                    langid=lang_prefix,
                    max_length=max_token,
                    use_rag=use_rag,
                    llm_first_token_callback=llm_first_token_callback,
                    llm_after_token_callback=llm_after_token_callback,
                    llm_model_name=llm_model_name,
                    text_out_callback=text_out_callback,
                    generate_audio=generate_audio,
                    load_model_start_callback=load_model_start_callback,
                    load_model_finish_callback=load_model_finish_callback,
                    error_callback=error_callback,
                )
                audio_thread.join()
            # 纯文字输出
            else:
                print("text ouput")
                chat(
                    prompt=prompt,
                    histories=histories,
                    langid=lang_prefix,
                    max_length=max_token,
                    use_rag=use_rag,
                    llm_first_token_callback=llm_first_token_callback,
                    llm_after_token_callback=llm_after_token_callback,
                    llm_model_name=llm_model_name,
                    text_out_callback=text_out_callback,
                    generate_audio=generate_audio,
                    load_model_start_callback=load_model_start_callback,
                    load_model_finish_callback=load_model_finish_callback,
                    error_callback=error_callback,
                )

            print("anwser finish!")
    finally:
        generating = False


def predict_llm_adapter(
    mode: int,
    model_name: str,
    input_data: str,
    histories: List[Dict[str, str]],
    generate_audio: bool,
    params: dict[str:any],
    ui_lang="zh_CN",
    use_rag=False,
    text_in_callback=None,
    text_out_callback=None,
    load_model_start_callback=None,
    load_model_finish_callback=None,
    sr_latency_callback=None,
    first_latency_callback=None,
    after_latency_callback=None,
    audio_out_callback=None,
    error_callback=None,
):
    global enable_memory_32g_feature
    max_token = params.get("max_token", 512)
    if mode == 1:
        predict(
            audio_input=input_data,
            histories=histories,
            text_in_callback=text_in_callback,
            text_out_callback=text_out_callback,
            llm_model_name=model_name,
            max_token=max_token,
            generate_audio=generate_audio,
            use_rag=use_rag,
            ui_lang=ui_lang,
            audio_out_callback=audio_out_callback,
            sr_latency_callback=sr_latency_callback,
            llm_first_token_callback=first_latency_callback,
            llm_after_token_callback=after_latency_callback,
            load_model_start_callback=load_model_start_callback,
            load_model_finish_callback=load_model_finish_callback,
            error_callback=error_callback,
        )
    else:
        if sr_latency_callback is not None:
            sr_latency_callback("")
        predict_user(
            prompt=input_data,
            histories=histories,
            generate_audio=generate_audio,
            llm_model_name=model_name,
            max_token=max_token,
            use_rag=use_rag,
            text_out_callback=text_out_callback,
            audio_out_callback=audio_out_callback,
            load_model_start_callback=load_model_start_callback,
            load_model_finish_callback=load_model_finish_callback,
            llm_first_token_callback=first_latency_callback,
            llm_after_token_callback=after_latency_callback,
            error_callback=error_callback,
        )


def predict(
    audio_input,
    histories: List[Dict[str, str]],
    text_in_callback=None,
    text_out_callback=None,
    llm_model_name="chatglm2-6b",
    max_token=512,
    use_rag=False,
    ui_lang="zh_CN",
    llm_first_token_callback=None,
    llm_after_token_callback=None,
    sr_latency_callback=None,
    generate_audio=True,
    audio_out_callback=None,
    load_model_start_callback=None,
    load_model_finish_callback=None,
    error_callback=None,
):  ## miss step
    import audioToText

    global sensitive_words_filter

    print("audio to text start")

    t0 = time.time()

    prompt_in = audioToText.get_long_prompt_funasr(audio_input)  ## 0301
    print(f'get prompt "{prompt_in}" from wav "{audio_input}"')
    t1 = time.time()
    sr_latency_count = (t1 - t0) * 1000

    print("sr_latency(ms): ", sr_latency_count)
    if sr_latency_callback is not None:
        sr_latency_callback(str(round(sr_latency_count, 2)) + " ms")

    input_lang = langid.classify(prompt_in)[0]  ## 0301

    print("------language: ", input_lang)  ## 1206

    if input_lang[:2] != "zh" and input_lang[:2] != "en":  ## 0131 0301
        print("This program only support Chinese and English!!!!!!!!!!!!!!!")  ## 1206

        if text_in_callback is not None:
            text_in_callback(
                "未能识别音频" if ui_lang == "zh_CN" else "Failure to recognize audio"
            )

        if text_out_callback is not None:
            text_out_callback(
                "对不起，没听清楚。请使用中文或英文复述一遍"
                if ui_lang == "zh_CN"
                else "sorry, I can't hear you. Please use Chinese or English Say it again"
            )

        if llm_first_token_callback is not None:
            llm_first_token_callback("")

        if llm_after_token_callback is not None:
            llm_after_token_callback("")

        raise customException.SpeechRecognitionException()
    elif sensitive_words_filter.exists(prompt_in):
        raise customException.SensitiveWordsException()
    else:
        if text_in_callback is not None:
            text_in_callback(prompt_in)

    predict_user(
        prompt=prompt_in,
        input_lang=input_lang,  # 0301
        histories=histories,
        text_out_callback=text_out_callback,
        llm_model_name=llm_model_name,
        max_token=max_token,
        use_rag=use_rag,
        llm_first_token_callback=llm_first_token_callback,
        llm_after_token_callback=llm_after_token_callback,
        generate_audio=generate_audio,
        audio_out_callback=audio_out_callback,
        load_model_start_callback=load_model_start_callback,
        load_model_finish_callback=load_model_finish_callback,
        error_callback=error_callback,
    )


def dispose():
    global chat_model, tts_executor, inited
    stop_generate()
    if chat_model is not None:
        try:
            del chat_model
            torch.xpu.empty_cache()
        except Exception:
            None
        finally:
            chat_model = None

    if tts_executor is not None:
        try:
            del tts_executor
        except Exception:
            None
        finally:
            tts_executor = None
    rag.dispose()
    gc.collect()
    inited = 0


def load_json_from_file(path: str):
    with open(path, mode="r", encoding="utf-8") as file:
        # 使用json.load方法加载JSON数据
        return json.load(file)


def get_init_settings():
    global debug_mode, model_path

    model_list = list()

    filenames = (
        ["aigc_setting_debug.json", "aigc_setting.json"]
        if debug_mode
        else ["aigc_setting.json"]
    )
    for name in os.listdir(model_path):
        # 在DEBUG模式下优先加载sd_cfg_debug.json，不存在时继续加载sd_cfg_.json
        for filename in filenames:
            cfg_path = os.path.join(model_path, name, filename)
            if os.path.exists(cfg_path):
                model_list.append(load_json_from_file(cfg_path))
                break

    return {"modelList": model_list}


# https://github.com/THUDM/ChatGLM2-6B/blob/main/web_demo2.py
# if __name__ == "__main__":
def model_list_f():
    model_list_all = list[str]()
    for model in get_init_settings().get("modelList"):
        model_list_all.append(model.get("model"))
    return model_list_all


inited = 0


def init_tts():
    global tts_executor
    if tts_executor is None:
        tts_executor = TTSExecutor()
        if not os.path.exists(save_wav_root_path1):
            os.makedirs(save_wav_root_path1)

        import web_utils

        temp_wav = web_utils.get_temp_filename(".wav")
        # 加载以下TTS模型，使后续使用速度加快
        load_tts_model_paddle("llm service start", temp_wav)
        os.remove(temp_wav)
        load_tts_model_paddle("大语言模型启动了", temp_wav)
        os.remove(temp_wav)


def unload_tts():
    global tts_executor
    if tts_executor is not None:
        del tts_executor
        tts_executor = None
        gc.collect()


def init(
    debug=False,
    model_find_path="./models/llm",
    save_wav_root_path="./static/audio_cache",
    memory_over32g=False,
):
    global \
        inited, \
        debug_mode, \
        model_path, \
        device, \
        model_load, \
        chat_model, \
        tokenizer, \
        model_loaded, \
        tts_executor, \
        chat_thread, \
        audio_thread, \
        user_stop_flag, \
        audio_generate_arg, \
        device_sr, \
        model_name_llm, \
        device_select_sr, \
        save_wav_root_path1, \
        enable_memory_32g_feature

    inited = 1
    debug_mode = debug
    model_path = model_find_path
    device = "None"
    device_sr = "iGPU"
    model_load = False
    chat_thread = None
    audio_thread = None
    user_stop_flag = False
    audio_generate_arg = AudioGenerateArg()
    tts_executor = None

    save_wav_root_path1 = save_wav_root_path

    device_select_sr = global_setup.device
    model_loaded = False

    model_name_llm = None
    print("loading tts fastspeech2_mix paddle ---------")
    chat_model, tokenizer = None, None  #### 1205

    # 30G以上内存才支持上历史下文问答
    enable_memory_32g_feature = memory_over32g

    inited = 2


class CustomStopCriteria(StoppingCriteria):
    """
    自定义停止条件
    ---------------
    ver: 2023-09-22
    by: changhongyu
    """

    def __init__(self, stop_callback):
        """
        :param token_id_list: 停止生成的指定token的id的列表
        """
        self.stop_callback = stop_callback

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        return self.stop_callback(input_ids, scores, **kwargs)


class AudioGenerateArg:
    segment_queue: Queue
    event: threading.Event
    out_finish: bool

    def __init__(self):
        """
        音频生成控制参数对象
        """
        self.segment_queue = Queue(-1)
        self.event = threading.Event()
        self.out_finish = False
