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
## 0304 add 、
## 0305 HuatuoGPT
#
# 音频文件需要时16k采样，转换为16k方法： ffmpeg -i input.wav -ar 16000 output.wav
# ffmpeg 切分16k音频方法： ffmpeg -i ~/whisper/chinese_5min3s.wav -ar 16000 -ss 00:00:00 -t 10s chinese_10s_16k.wav

# gradio UI loading 太慢 https://github.com/gradio-app/gradio/issues/4332#issuecomment-1563758104
# vim ~/miniconda3/envs/llm/lib/python3.9/site-packages/gradio/themes/utils/fonts.py # 注意需要修改一下虚拟环境的名字
# 大约在第50行的位置，注释return 那一行，并且在前面加pass
# def stylesheet(self) -> str:
#     pass
#     #return f'https://fonts.googleapis.com/css2?family={self.name.replace(" ", "+")}:wght@{";".join(str(weight) for weight in self.weights)}&display=swap'

import os
import torch
import time
import sys
import traceback
# import argparse
import numpy as np
import scipy

from ipex_llm.transformers import AutoModelForCausalLM
from ipex_llm.transformers import AutoModelForSpeechSeq2Seq

# from transformers import AutoModelForSpeechSeq2Seq

# from transformers import LlamaTokenizer
import intel_extension_for_pytorch as ipex
from transformers import WhisperProcessor

# from transformers import TextStreamer,TextIteratorStreamer
# from colorama import Fore
# import speech_recognition as sr
from ipex_llm.transformers import AutoModel
from transformers import AutoTokenizer

# import gradio as gr
# streamlit as st
# from streamlit_chat import message
import gc
import re

# import mdtex2html
# import intel_extension_for_pytorch as ipex
# import soundfile
# import pyaudio
# import wave
# import TTS
import paddle
from paddlespeech.cli.tts import TTSExecutor
import shutil
import speech_recognition as sr

# from audio_recorder_streamlit import audio_recorder
# from TTS.api import TTS
# from playsound import playsound
# import asyncio
# import edge_tts
from threading import Thread
from multiprocessing import Queue
import pyaudio
import wave
from datetime import datetime
import soundfile
import gradio as gr
import mdtex2html
import langid
from zhconv import convert
from transformers import TextIteratorStreamer
from timer import timer
import transformers
from transformers.tools.agents import StopSequenceCriteria
# import onnxruntime as ort
# from paddlespeech.t2s.frontend.mix_frontend import MixFrontend
# from mix_frontend import MixFrontend
from paddlespeech.t2s.exps.syn_utils import run_frontend

# from openvino.runtime import Core
# from stablediffusionOV import ImageGenerator
# from translate import Translator
import psutil

# from bark import SAMPLE_RATE, generate_audio, preload_models
# from scipy.io.wavfile import write as write_wav
# from FastSpeech2.speech import get_speech

os.environ["USE_XETLA"] = "OFF"
os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
os.environ["ENABLE_SDP_FUSION"] = "1"

# device_sd = "gpu.1"
# device_whisper = "xpu"
# you could tune the prompt based on your own model,
#CHATGLM_V3_PROMPT_FORMAT = "<|system|>\nYou are an intelligent AI assistant, named ChatGLM3. Follow the user's instructions carefully.\n<|user|>\n{prompt}\n<|assistant|>\n"  ## 1220
CHATGLM_V3_PROMPT_FORMAT = "问：{prompt}\n\n答：\n" ## 0111
CHATGLM_V3_PROMPT_FORMAT_en = "Question: {prompt}\nAnswer: \n" # 0301
#CHATGLM_V3_PROMPT_FORMAT = "{prompt}" ## 0111
#CHATGLM_V3_PROMPT_FORMAT_en = "{prompt}"
#CHATGLM_V3_PROMPT_FORMAT_en = "<|system|>\nYou are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.\n<|user|>\n{prompt}\n<|assistant|>\n"
QWEN_PROMPT_FORMAT = "<|im_start|>system\n助理 是阿里巴巴训练的大型语言模型。\n<|im_end|>\n<|im_start|>用户\n{prompt}\n<|im_end|>\n<|im_start|>助理\n"   ## 1220
CHATGLM_V2_PROMPT_FORMAT_zh = "问：{prompt}\n\n答：\n"  ### 1205 chatglm2 and chatglm3
CHATGLM_V2_PROMPT_FORMAT_en = "Question:{prompt}\nAnswer:\n"  ### 1205
BAICHUAN2_PROMPT_FORMAT = "问：{prompt} 答：\n"  ### 1205
BAICHUAN2_PROMPT_FORMAT_en = "Question: {prompt}\nAnswer: \n" # 0301
INTERNLM_PROMPT_FORMAT = "<|User|>:{prompt}\n<|Bot|>:\n"  ### 1205
AQUILA2_PROMPT_FORMAT = "<|startofpiece|>{prompt}\n<|endofpiece|>\n"  ### 1205
HUATUOGPT_PROMPT_FORMAT = "<病人>：{prompt}\n <HuatuoGPT>：" # 0305

chatglm2_model:transformers.PreTrainedModel
# check if cl_cache exist or not
cache_dir = os.path.expanduser("~/cl_cache")

if not os.path.exists(cache_dir):
    # create the folder if it does not exist
    os.makedirs(cache_dir)


def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)

    info = p.memory_full_info()
    memory = info.uss / 1024.0 / 1024
    print("******************* {} memory used: {} MB".format(hint, memory))


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


def resample(audio, src_sample_rate, dst_sample_rate):
    """
    Resample audio to specific sample rate

    Parameters:
      audio: input audio signal
      src_sample_rate: source audio sample rate
      dst_sample_rate: destination audio sample rate
    Returns:
      resampled_audio: input audio signal resampled with dst_sample_rate
    """
    if src_sample_rate == dst_sample_rate:
        return audio
    duration = audio.shape[0] / src_sample_rate
    resampled_data = np.zeros(shape=(int(duration * dst_sample_rate)), dtype=np.float32)
    x_old = np.linspace(0, duration, audio.shape[0], dtype=np.float32)
    x_new = np.linspace(0, duration, resampled_data.shape[0], dtype=np.float32)
    resampled_audio = np.interp(x_new, x_old, audio)
    return resampled_audio.astype(np.float32)


def get_input_features2(processor, audio_file, device):
    sample_rate = audio_file[0]
    audio = audio_file[1]
    audio = (
        audio.astype(np.float32) / np.iinfo(audio.dtype).max
    )  # 除以65536, 之前demo 是除以32768.0
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        print("=====rasample audio to 16000Hz")
        audio = resample(audio, sample_rate, 16000)

    input_features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features
    if device == "xpu":
        input_features = input_features.half().contiguous().to(device)
    # else:
    #    input_features = input_features.contiguous().to(device)
    return input_features


def get_input_features(processor, audio_file, device):
    with sr.AudioFile(audio_file) as source:
        audio = sr.Recognizer().record(source)  # read the entire audio file
    frame_data = (
        np.frombuffer(audio.frame_data, np.int16).flatten().astype(np.float32) / 32768.0
    )
    # audio.ndim == 2:
    #    audio = audio.mean(axis=1)
    if audio.sample_rate != 16000:
        print("=====rasample audio to 16000Hz")
        frame_data = resample(frame_data, audio.sample_rate, 16000)
    input_features = processor(
        frame_data, sampling_rate=16000, return_tensors="pt"
    ).input_features
    if device == "xpu":
        # input_features = input_features.half().contiguous().to(device)
        input_features = input_features.contiguous().to(device)
    return input_features

## 0124 start
from funasr_onnx import Paraformer,CT_Transformer
def load_asr_model(model_path):
    asr_model_path = model_path + "paraformer-large"
    print("loading paraformer-large---------")
    t0 = time.time()
    asr_model = Paraformer(asr_model_path, batch_size=1, quantize=True)
    t1 = time.time()
    print("loading paraformer-large----------Done, cost time(s): ", t1 - t0)
    ct_model_path  = model_path + "ct-transformer-large"
    print("loading ct-transformer-large---------")
    t0 = time.time()
    ct_model = CT_Transformer(ct_model_path, batch_size=1, quantize=True)
    t1 = time.time()
    print("loading ct-transformer-large----------Done, cost time(s): ", t1 - t0)
    return asr_model, ct_model

def get_prompt_funasr(asr_model, ct_model, audio, device="cpu"):
    with torch.inference_mode():
        print("-----audio: ", audio)
        result = asr_model(audio)# TO DO: aother sample rate
        output_str = ct_model(text=result[0]['preds'][0])[0]
    return output_str,result[0]["preds"][0] # 0301

## 0124 end

def load_whisper_model_cpu(model_path, device="cpu"):
    # whisper_model_path = model_path + "/../whisper-small/"
    print("loading whisper---------")
    t0 = time.time()

    import whisper as WHisper

    model = WHisper.load_model("small")
    end = time.time()

    if 0:
        # from bigdl.llm import optimize_model
        import librosa

        y, sr = librosa.load("hongqiao.wav")

        # Downsample the audio to 16kHz
        audio = librosa.resample(y, orig_sr=sr, target_sr=16000)

        model = WHisper.load_model("small")

        # model = optimize_model(model)
        st = time.time()
        result = model.transcribe(audio)
        end = time.time()
        print(f"whisper Inference time: {end-st} s")  ## 8s use 13s
        print(result["text"])

    print("loading whisper----------Done, cost time(s): ", end - t0)
    return model


def load_whisper_model(model_path, sr_model, device):
    whisper_model_path = model_path + sr_model + "-int4/"
    print("loading whisper---------", whisper_model_path)
    t0 = time.time()

    whisper = AutoModelForSpeechSeq2Seq.load_low_bit(
        whisper_model_path,
        trust_remote_code=True,
        optimize_model=False,
        tie_word_embeddings=False,
    )
    processor = WhisperProcessor.from_pretrained(whisper_model_path)
    # whisper =  AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_path, trust_remote_code=True,load_in_4bit=True)
    whisper.config.forced_decoder_ids = None
    # whisper = whisper.half().to(device)
    whisper = whisper.to(device)

    # if device == "xpu":
    ## model.model.encoder.embed_positions.to("cpu")
    #    model.model.decoder.embed_tokens.to("cpu")
    # whisper = BenchmarkWrapper(whisper, do_print=True)
    t1 = time.time()
    print("loading whisper----------Done, cost time(s): ", t1 - t0)

    # input_features = get_inputchatglm2_model_features(processor, "hongqiao.wav", device)#0.09s
    # print(" 2 ")
    # print("input_features",input_features)
    # predicted_ids = whisper.generate(input_features)
    # print(" 3 ")
    # output_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    # print("output_str",output_str)

    return processor, whisper


def load_chatglm2_model(model_path, model_name_llm, device):
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
        tokenizer = AutoTokenizer.from_pretrained(
            chatglm2_model_path, trust_remote_code=True
        )
    else:
        chatglm2_model = AutoModelForCausalLM.load_low_bit(
            chatglm2_model_path,
            trust_remote_code=True,
            optimize_model=True,
            use_cache=True,
            replace_embedding=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            chatglm2_model_path, trust_remote_code=True
        )

    # chatglm2_model =  AutoModel.load_low_bit(chatglm2_model_path, trust_remote_code=True, optimize_model=True,use_cache=True,replace_embedding=True).eval()
    # chatglm2_model = AutoModel.from_pretrained(chatglm2_model_path, trust_remote_code=True, optimize_model=True, load_in_4bit=True).eval()
    chatglm2_model.to(device)

    ## for MTL iGPU
    # if device == "xpu":
    #    chatglm2_model.transformer.embedding.to('cpu')

    tokenizer = AutoTokenizer.from_pretrained(
        chatglm2_model_path, trust_remote_code=True
    )
    torch.xpu.synchronize()
    t3 = time.time()
    print("loading LLM---------Done, cost time(s): ", chatglm2_model_path, t3 - t2)
    return chatglm2_model, tokenizer


def load_tts_model_paddle(text_in="今天的天气不错啊", audio_out="./output.wav"):
    #  print("loading tts fastspeech2_mix paddle ---------")
    #  tts_executor = TTSExecutor()
    wav_file = tts_executor(
        text=text_in,
        output=audio_out,
        am="fastspeech2_mix",
        am_config=None,
        am_ckpt=None,
        am_stat=None,
        spk_id=174,
        phones_dict=None,
        tones_dict=None,
        speaker_dict=None,
        voc="hifigan_csmsc",
        voc_config=None,
        voc_ckpt=None,
        voc_stat=None,
        lang="mix",
        device="cpu",
    )
    print("***********Wave file has been generated: {}".format(wav_file))
    # return wav_file


def load_tts_model2(model_path, device):
    print("loading tts fastspeech2_mix---------")
    show_memory_info("before loading tts ")
    t4 = time.time()
    cpu_threads = 4
    spk_id = 174

    # am = 'fastspeech2_mix'
    phones_dict = model_path + "fastspeech2_mix_onnx_0.2.0/phone_id_map.txt"
    am_model_path = model_path + "fastspeech2_mix_onnx_0.2.0/fastspeech2_mix.onnx"
    voc_model_path = model_path + "fastspeech2_mix_onnx_0.2.0/hifigan_csmsc.onnx"
    show_memory_info("before loading tts 1 ")

    tts_frontend = MixFrontend(phone_vocab_path=phones_dict)
    show_memory_info("before loading tts 2 ")
    providers = ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()

    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    show_memory_info("before loading tts 3 ")
    sess_options.intra_op_num_threads = cpu_threads

    am_sess = ort.InferenceSession(
        am_model_path, providers=providers, sess_options=sess_options
    )

    voc_sess = ort.InferenceSession(
        voc_model_path, providers=providers, sess_options=sess_options
    )
    print("tts fastspeech2_mix load model done! Warmup start----")
    merge_sentences = True

    # from paddlespeech.cli.tts.infer import TTSExecutor
    # tts = TTSExecutor()
    # tts(text="今天天气十分不错。", output="output.wav")

    # frontend warmup
    # Loading model cost 0.5+ seconds
    tts_frontend.get_input_ids(
        "hello, thank you, thank you very much", merge_sentences=merge_sentences
    )
    print("tts fastspeech2_mix load model done! Warmup start  am warmup ----")
    # am warmup
    spk_id = [spk_id]
    for T in [27, 38, 54]:
        am_input_feed = {}
        phone_ids = np.random.randint(1, 266, size=(T,))
        am_input_feed.update({"text": phone_ids})
        am_input_feed.update({"spk_id": spk_id})
        print(" am warmup 1----")
        am_sess.run(None, input_feed=am_input_feed)
        print(" am warmup 2----")
    print("tts fastspeech2_mix load model done! Warmup start  voc warmup ----")
    # voc warmup
    for T in [227, 308, 544]:
        data = np.random.rand(T, 80).astype("float32")
        voc_sess.run(None, input_feed={"logmel": data})
    print("tts warm up done!")
    t5 = time.time()
    print("loading TTS fastspeech2_mix---------Done, cost time(s): ", t5 - t4)

    print("loading TTS tacotron2-DDC---------")
    # tts_en_model_path = model_path + "tacotron2-DDC/"
    # t6 = time.time()
    # tts_en_model = TTS(model_path=tts_en_model_path+"model_file.pth", config_path=tts_en_model_path+"config.json", progress_bar=True, gpu=False) #0.5s
    # t7 = time.time()
    print("loading TTS en---------Done, cost time(s): ", t7 - t6)

    # return tts_frontend, am_sess, voc_sess, spk_id, tts_en_model
    return tts_frontend, am_sess, voc_sess, spk_id, None


def load_tts_model(model_path, device):
    print("loading TTS tacotron2-DDC-GST---------")
    tts_model_path = model_path + "tacotron2-DDC-GST/"
    tts_en_model_path = model_path + "tacotron2-DDC/"
    t4 = time.time()
    tts_model = TTS(
        model_path=tts_model_path + "model_file.pth",
        config_path=tts_model_path + "config.json",
        progress_bar=True,
        gpu=False,
    )  # 0.5s
    t5 = time.time()
    print("loading TTS zh---------Done, cost time(s): ", t5 - t4)

    print("loading TTS tacotron2-DDC---------")
    t6 = time.time()
    tts_en_model = TTS(
        model_path=tts_en_model_path + "model_file.pth",
        config_path=tts_en_model_path + "config.json",
        progress_bar=True,
        gpu=False,
    )  # 0.5s
    t7 = time.time()
    print("loading TTS en---------Done, cost time(s): ", t7 - t6)
    return tts_model, tts_en_model


def load_model(model_path, device, model_loaded):
    # try:
    print(
        "******************** Loading Whisper-medium & ChatGLM2 from "
        + model_path
        + " to "
        + device
        + " ************************"
    )
    whisper_model_path = model_path + "whisper-medium-int4"
    chatglm2_model_path = model_path + "chatglm2-6b-int4"
    # print("whisper_model_path:", whisper_model_path)
    # device == "xpu" and model_loaded is False:
    print("loading whisper---------")
    t0 = time.time()
    processor = WhisperProcessor.from_pretrained(whisper_model_path)
    whisper = AutoModelForSpeechSeq2Seq.load_low_bit(
        whisper_model_path, trust_remote_code=True, optimize_model=False
    )
    # whisper =  AutoModelForSpeechSeq2Seq.load_low_bit(whisper_model_path, trust_remote_code=True)
    whisper.config.forced_decoder_ids = None
    whisper = whisper.half().to(device)
    t1 = time.time()
    print("loading whisper----------Done, cost time(s): ", t1 - t0)

    # input_features = get_input_features(processor, "hongqiao.wav", device)#0.09s
    print(" 2 ")
    print("input_features", input_features)
    predicted_ids = whisper.generate(input_features)
    print(" 3 ")
    output_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print("output_str", output_str)

    print("loading chatglm2---------")
    t2 = time.time()
    chatglm2_model = AutoModel.load_low_bit(
        chatglm2_model_path, trust_remote_code=True, optimize_model=False
    )
    # chatglm2_model = chatglm2_model.half().to('xpu')
    chatglm2_model = chatglm2_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        chatglm2_model_path, trust_remote_code=True
    )
    t3 = time.time()
    print("loading chatglm2---------Done, cost time(s): ", t3 - t2)

    # print("loading TTS tacotron2-DDC-GST---------")
    # t4 = time.time()
    # tts_model = TTS(model_path="./tacotron2-DDC-GST/model_file.pth", config_path="./tacotron2-DDC-GST/config.json", progress_bar=True, gpu=False) #0.5s
    # t5 = time.time()
    # print("loading TTS zh---------Done, cost time(s): ", t5-t4)

    # print("loading TTS tacotron2-DDC---------")
    # t6 = time.time()
    # tts_en_model = TTS(model_path="./tacotron2-DDC/model_file.pth", config_path="./tacotron2-DDC/config.json", progress_bar=True, gpu=False) #0.5s
    # t7 = time.time()
    # print("loading TTS en---------Done, cost time(s): ", t7-t6)

    print("=========================total load time(s): ", t5 - t4 + t3 - t2 + t1 - t0)
    model_loaded = True
    #   return whisper, processor, chatglm2_model, tokenizer, tts_model, tts_en_model, model_loaded
    return whisper, processor, chatglm2_model, tokenizer, None, None, model_loaded
    # except:
    #    print("******************** Can't find local model\n exit ************************")
    #    sys.exit(1)


def get_prompt(processor, whisper, audio, device):
    with torch.inference_mode():
        print(" 1 ")
        input_features = get_input_features(processor, audio, device)  # 0.09s
        print("SR device ", device)
        print("input_features", input_features)
        predicted_ids = whisper.generate(input_features)
        print(" 3 ")
        output_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print("output_str", output_str)
    return output_str


def save_wav(*, wav: list, path: str, sample_rate: int = 22050, **kwargs) -> None:
    """Save float waveform to a file using Scipy.

    Args:
        wav (np.ndarray): Waveform with float values in range [-1, 1] to save.
        path (str): Path to a output file.
        sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
    """
    wav = np.array(wav)
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    scipy.io.wavfile.write(path, sample_rate, wav_norm.astype(np.int16))


# async def get_tts(text, audio_file) -> None:
#    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
#    await communicate.save(audio_file)


def get_tts2(text, audio_file):
    text = text.replace("\n", ",")
    text = text.replace(" ", "")
    text = text + "。"
    # tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST", progress_bar=True, gpu=False)    tt1 = time.time()
    tts_model.tts_to_file(text, file_path=audio_file)


def get_tts(text, audio_file):
    text = text.replace("\n", ",")
    text = text.replace(" ", "")
    text = text + "。"
    # tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST", progress_bar=True, gpu=False)    tt1 = time.time()
    new_text = ""
    start = 0
    temp = re.split("。|！|？", text)
    for i in range(len(temp) - 1):
        if len(temp[i]) > 50:
            print("======sentence len > 50, convert!!!", len(temp[i]), temp[i])
            temp[i] = temp[i].replace(",", "。")
        new_text += temp[i] + text[start + len(temp[i])]
        start = len(new_text)
    tts_model.tts_to_file(new_text, file_path=audio_file)
    # tts_model.tts_to_file(text, file_path=audio_file)
    # tts_model.tts_to_file(text.replace("\n",",").replace(" ","")+"。", file_path=audio_file)


def get_tts_en(text, audio_file):
    text = text.replace("\n", ",")
    tts_en_model.tts_to_file(text, file_path=audio_file)


def get_tts3(text, audio_file):
    fs = 24000
    am_input_feed = {}
    with timer() as t:
        frontend_dict = run_frontend(
            frontend=tts_frontend,
            text=text,
            merge_sentences=False,
            get_tone_ids=False,
            lang="mix",
        )
        phone_ids = frontend_dict["phone_ids"]
        flags = 0
        # if len(phone_ids) == 0:
        #    print("len(phone_ids)-----------", len(phone_ids), phone_ids, "\n text: ", text, "\n len text: ", len(text))
        for i in range(len(phone_ids)):
            part_phone_ids = phone_ids[i].numpy()
            am_input_feed.update({"text": part_phone_ids})
            am_input_feed.update({"spk_id": spk_id})
            mel = am_sess.run(output_names=None, input_feed=am_input_feed)
            mel = mel[0]
            wav = voc_sess.run(output_names=None, input_feed={"logmel": mel})
            wav = wav[0]
            if flags == 0:
                wav_all = wav
                flags = 1
            else:
                wav_all = np.concatenate([wav_all, wav])
    # if len(phone_ids) == 0:
    #    return
    # else:
    wav = wav_all
    speed = len(wav) / t.elapse
    rtf = fs / speed
    soundfile.write(audio_file, wav, samplerate=fs)
    print(
        f"mel: {mel.shape}, wave: {len(wav)}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
    )

 ## 1220 start
from transformers.generation.stopping_criteria import StoppingCriteria
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = []):
        StoppingCriteria.__init__(self)

    def __call__(self,input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
        self.stops = stops
        for i in range(len(stops)):
            self.stops = self.stops[i]

def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == "Qwen":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id], [tokenizer.eod_id]]
    elif chat_format == "ChatGLM3":
        stop_words_ids = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), tokenizer.get_command("<|observation|>"), tokenizer.get_command("<|assistant|>")]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids      
 ## 1220 end

def stream_chat_generate(model:transformers.PreTrainedModel, args:dict, error_callback=None):
    
    try:
        model.generate(**args)
    except Exception as ex:
        traceback.print_exc()
        if error_callback != None:
            error_callback(ex)

## 0305 start 
def generate_prompt(query, history, eos):
    if len(history) != None:
        return f"""一位用户和智能医疗大模型HuatuoGPT之间的对话。对于用户的医疗问诊，HuatuoGPT给出准确的、详细的、温暖的指导建议。对于用户的指令问题，HuatuoGPT给出有益的、详细的、有礼貌的回答。<病人>：{query} <HuatuoGPT>："""
    else:
        prompt = '一位用户和智能医疗大模型HuatuoGPT之间的对话。对于用户的医疗问诊，HuatuoGPT给出准确的、详细的、温暖的指导建议。对于用户的指令问题，HuatuoGPT给出有益的、详细的、有礼貌的回答。'
        for i, (old_query, response) in enumerate(history):
            prompt += "<病人>：{} <HuatuoGPT>：{}".format(old_query, response) + eos
        prompt += "<病人>：{} <HuatuoGPT>：".format(query)
        return prompt     

def clear_history():
    global history,history_round
    history = []  
    history_round == 0
## 0305 end

def stream_chat(model:transformers.PreTrainedModel, tokenizer:AutoTokenizer, prompt:str, max_new_tokens:int, device="xpu", error_callback=None):
    global chat_thread, user_stop_flag, model_name_llm
    if model_name_llm == "HuatuoGPT-7B":
        temperature=0.2
        repetition_penalty=1.2
        context_len=1024
        stream_interval=1

        prompt = generate_prompt(prompt, history, tokenizer.eos_token)
        input_ids = tokenizer(prompt).input_ids
        output_ids = list(input_ids)

        device = model.device
        stop_str = tokenizer.eos_token
        stop_token_ids = [tokenizer.eos_token_id]

        l_prompt = len(tokenizer.decode(input_ids, skip_special_tokens=False))

        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        pre = 0 # 0305
        for i in range(max_new_tokens):
            if i == 0:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]

            if device == "mps":
                # Switch to CPU by avoiding some bugs in mps backend.
                last_token_logits = last_token_logits.float().to("cpu")

            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                output = tokenizer.decode(output_ids, skip_special_tokens=False)
               ## print(output)
               # print(l_prompt, i, max_new_tokens)
                if stop_str:
                    pos = output.rfind(stop_str, l_prompt)
                  #  print(pos)
                    if pos != -1:
                        output = output[l_prompt:pos]
                        stopped = True
                    else:
                        output = output[l_prompt+pre:]  # 0305
                        pre += len(output) 
                       # print("*****",i,pre,output)
                        yield output
                else:
                    raise NotImplementedError

            if stopped:
                break
        del past_key_values

    else:
        input_ids = tokenizer([prompt], return_tensors="pt").to(device)
        print("stream_chat-----input_ids:", input_ids)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,  # skip prompt in the generated tokens
            skip_special_tokens=True,
        )

        def user_stop(input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
            global user_stop_flag
            return user_stop_flag

        stopping_criteria = transformers.StoppingCriteriaList()   ## 1220
        if model_name_llm == "Qwen-7B-Chat":
            stopping_criteria.append(StoppingCriteriaSub(stops = get_stop_words_ids("Qwen", tokenizer=tokenizer)))
        elif model_name_llm == "chatglm3-6b":
            print("tokenizer.eos_token_id: ", tokenizer.eos_token_id, tokenizer.tokenize("<eop>"), tokenizer.get_command("<|user|>"), tokenizer.get_command("<|observation|>"))
            # stopping_criteria.append(StopSequenceCriteria(["<|user|>","<|observation|>","<|assistant|>", "<eop>", '▁<eop>'], tokenizer))
            stopping_criteria.append(StoppingCriteriaSub(stops = get_stop_words_ids("ChatGLM3", tokenizer=tokenizer)))
        stopping_criteria.append(CustomStopCriteria(user_stop))

        generate_kwargs = dict(
            input_ids,
            streamer=streamer,
            num_beams=1,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
        )

        chat_thread = threading.Thread(target=stream_chat_generate,args=(model,generate_kwargs,error_callback))
        chat_thread.start()
    
        return streamer


def chat(
    prompt: str,
    max_length: int,
    top_p,
    temperature,
    llm_first_token_callback,
    llm_after_token_callback,
    llm_model_name,
    text_out_callback,
    generate_audio: bool,
    load_model_start_callback,
    load_model_finish_callback,
    error_callback = None
):
    global chatglm2_model, tokenizer, model_path, model_name_llm, audio_generate_arg, user_stop_flag, history, history_round,max_len_history
    count = 0
    tmp = 2
    all_flag = True
    rest_count = 0

    if llm_model_name != model_name_llm:
        print("new model--------------restart a new chat")
        history_round = 0
        history = []
        tokenizer = None  ##  0109
    if model_name_llm == "chatglm3-6b":
        max_round = 5
    else:
        max_round = 3
    if llm_model_name != "HuatuoGPT-7B":
        if tokenizer:
            if len(tokenizer.tokenize(history)) > 2000 or history_round > max_round:
                print("!!!!!!!!! history len={} > 2k or round={} > {}--------------restart a new chat".format(len(tokenizer.tokenize(history)), history_round, max_round))
            # history_round = 0
            # history = []  
                history = history.split("\n问：",1)[-1] ## 0118
        else:
            history_round = 0
            history = []

    if llm_model_name == "HuatuoGPT-7B":# 0305
        if history_round == 0:
            history = []
        if history_round > max_round:
           ## if max_len_history > 2000:
           #     history = history[max_round:]  
           # else:
            history = history[1:]


        print("____ HuatuoGPT-7B history",history) # 0305
        
    else:
        if len(history) == 0 : 
            print("--------------------- new chat ")
            prompt = prompt
            history = prompt         
        else:
            prompt = history + "\n" + prompt
    #print(history)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ round=", history_round)
    print("--------------------------------------history + prompt:\n{}\n---------------------------------------\n".format(prompt))
    history_round += 1

    if llm_model_name != model_name_llm:
        if chatglm2_model != None:
            chatglm2_model.to("cpu")
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
            del chatglm2_model
            gc.collect()

        model_name_llm = llm_model_name
        model_full_path = model_path + model_name_llm + "-int4"
        print("******* loading ", model_name_llm)
        if load_model_start_callback != None:
            load_model_start_callback(model_name_llm)
        stm = time.time()

        if model_name_llm == "chatglm2-6b" or model_name_llm == "chatglm3-6b":
            # chatglm2_model = AutoModel.from_pretrained(model_full_path, trust_remote_code=True, optimize_model=True, load_in_4bit=True).eval()
            chatglm2_model = AutoModel.load_low_bit(
                model_full_path,
                trust_remote_code=True,
                optimize_model=True,
                use_cache=True,
                replace_embedding=True,
            ).eval()
            tokenizer = AutoTokenizer.from_pretrained(
                model_full_path, trust_remote_code=True
            )
            chatglm2_model.to("xpu")

        elif model_name_llm == "internlm-chat-20b":
            chatglm2_model = AutoModelForCausalLM.load_low_bit(
                model_full_path,
                trust_remote_code=True,
                optimize_model=True,
                use_cache=True,
                replace_embedding=True,
            ).eval()
            tokenizer = AutoTokenizer.from_pretrained(
                model_full_path, trust_remote_code=True
            )
            chatglm2_model.to("xpu")
        else:
            chatglm2_model = AutoModelForCausalLM.load_low_bit(
                model_full_path,
                trust_remote_code=True,
                optimize_model=True,
                use_cache=True,
                replace_embedding=True,
            ).eval()
            #  model = BenchmarkWrapper(model)).eval()
            tokenizer = AutoTokenizer.from_pretrained(
                model_full_path, trust_remote_code=True
            )
            chatglm2_model.to("xpu")
        print("********** model load time (s)= ", time.time() - stm)
        if load_model_finish_callback != None:
            load_model_finish_callback(model_name_llm)

    torch.xpu.synchronize()
    timeStart = time.time()
    timeFirstRecord = False
    with torch.inference_mode():
        response = ""
        for stream_output in stream_chat(
            chatglm2_model, tokenizer, prompt, max_length, "xpu", error_callback
        ):
            # for response, history  in chatglm2_model.stream_chat(tokenizer, prompt,
            #                                         history, max_length=max_length, top_p=top_p,
            #                                         temperature=temperature):
            # print(f"chat response {response}")
            if user_stop_flag:
                print("chat exit by user stop")
                return
            if text_out_callback != None:
                text_out_callback(stream_output)
            #print("stream_output",stream_output)
            response += stream_output
            # response_queue.append(response)
            # message_placeholder.markdown(response)
            # Add a blinking cursor to simulate typing
            if timeFirstRecord == False:
                torch.xpu.synchronize()
                timeFirst = time.time()  # - timeStart
                timeFirstRecord = True
                print(f"===============Get first token at {datetime.now()}")

            token_num = len(tokenizer.tokenize(response))
            if token_num > 300 and all_flag: # 0301
                all_flag = False
            elif generate_audio:
                # rest_count = count
                if (
                    (
                        "。" in response[rest_count:]
                        or "！" in response[rest_count:]
                        or "？" in response[rest_count:]
                        or "：" in response[rest_count:]
                        or "." in response[rest_count:]
                        or "!" in response[rest_count:]
                        or "?" in response[rest_count:]
                        or ":" in response[rest_count:]
                    )
                    and token_num > 40
                    and all_flag
                ):
                    segment = re.split("[。！？：.!?:]", response[rest_count:])
                    # print("1-------------segment: ", segment)
                    if len(segment) > 1:
                        audio_generate_arg.segment_queue.put(segment[-2])
                        audio_generate_arg.event.set()
                        rest_count += len(segment[-2]) + 1

                elif (
                    (
                        "。" in response[rest_count:]
                        or "！" in response[rest_count:]
                        or "，" in response[rest_count:]
                        or "？" in response[rest_count:]
                        or "：" in response[rest_count:]
                        or "," in response[rest_count:]
                        or "." in response[rest_count:]
                        or "!" in response[rest_count:]
                        or "?" in response[rest_count:]
                        or ":" in response[rest_count:]
                        or "、" in response[rest_count:] # 0304
                    )
                    and token_num <= 40
                    and all_flag
                ):
                    segment = re.split("[。！？：.!?:，,、]", response[rest_count:]) # 0304
                    print("2-------------segment: ", segment,len(segment))
                    if len(segment) > 1:
                        audio_generate_arg.segment_queue.put(segment[-2])
                        audio_generate_arg.event.set()
                        rest_count += len(segment[-2]) + 1
                # elif token_num == 1 and all_flag:    
                #     segment = re.split("[。！？：.!?:，,]", response[rest_count:])  
                #     print("33-------------segment: ", segment,len(segment)) 
                #     if len(segment) == 1 and segment != ['']:      
                #         print("44-------------segment: ", segment)                                 ### 1220
                #         audio_generate_arg.segment_queue.put(segment[0])  ### 1220
                #         audio_generate_arg.event.set()                    ### 1220
                #         rest_count += len(segment[0]) + 1
                      
        timeTotal = time.time()
        llm_time = timeTotal - timeStart
        token_count_input = len(tokenizer.tokenize(prompt))
        token_count_output = len(tokenizer.tokenize(response))
        llm_ms_first_token = (timeFirst - timeStart) * 1000
        if token_count_output == 0 or token_count_output == 1:  ## 1220
            llm_ms_after_token = ((timeTotal - timeFirst) * 1000) ### 1220
        elif token_count_output > 1:
            llm_ms_after_token = (
                (timeTotal - timeFirst) / (token_count_output - 1) * 1000
            )

        if generate_audio:
            segment = response[rest_count:]
            audio_generate_arg.segment_queue.put(segment)
            audio_generate_arg.event.set()

    print("Prompt: ", prompt)
    print("Response: ", response)
    print("token count input: ", token_count_input)
    print("token count output: ", token_count_output)
    print("LLM First token latency(ms): ", llm_ms_first_token)
    print("LLM After token latency(ms/token): ", llm_ms_after_token)
    if model_name_llm == "HuatuoGPT-7B":
        history = history + [(prompt, response)]
        max_len_history += token_count_input+token_count_output
        print("******** max_length history", max_len_history)
    else:
        history = prompt + response
        print("******** max_length history", len(tokenizer.tokenize(history)))
    print("LLM time cost(s): ", llm_time)

    # llm_ms_first_token_list.append(llm_ms_first_token)
    # llm_ms_after_token_list.append(llm_ms_after_token)
    if llm_first_token_callback != None:
        llm_first_token_callback(f"{round(llm_ms_first_token, 2)} ms")
    if llm_after_token_callback != None:
        llm_after_token_callback(f"{round(llm_ms_after_token, 2)} ms/token")
    print("-" * 50)
    print("\n")


#  return llm_ms_first_token_list,llm_ms_after_token_list



def get_speech(finish_flag: list[bool], lang: str, wav_path_callback):
    global audio_generate_arg, save_wav_root_path1
    audio_folder = os.path.join(os.getcwd(), save_wav_root_path1)
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)
    while True:
        while not user_stop_flag and not audio_generate_arg.segment_queue.empty():
            text = audio_generate_arg.segment_queue.get_nowait()
            if not user_stop_flag and text is not None and text != "":
                print("text in get_speech: ", text)
                audio_out = f"{audio_folder}/{str(datetime.now()).replace(':', '-').replace(' ','_')}.mp3"
                # asyncio.run(get_tts(text, audio_out))
                text = text.replace("ChatGLM2-6B", "chat G L M 2 - 6 B")
                text = text.replace("AI", "A.I")
                for word in re.findall("\s[A-Z]+|[A-Z]+\s", text):  # 往大写字母单词间加空格
                    new_word = re.sub(r"(?<=\w)(?=\w)", " ", word)
                    text = re.sub(word, new_word, text)
                if lang == "zh" or lang == "en" or lang == "zh_CN" or lang == "en_US":    ### 1220
                    #  get_tts3(text, audio_out)
                    load_tts_model_paddle(text, audio_out)
                else:
                    # get_tts_en(text, audio_out)
                    #  get_tts3(text, audio_out)
                    print("The lang is ",lang,"!!! TTS only support Chinese and English !!!")  ### 1220
                    load_tts_model_paddle(text, audio_out)
                #         data, samplerate = soundfile.read(audio_out) #file does not start with RIFF id
                #         start = round(np.nonzero(np.array(data))[0].tolist()[0] * 0.95)
                #     # end = round(np.nonzero(np.array(data))[0].tolist()[-1] * 1.05)
                #         end = round(np.nonzero(np.array(data))[0].tolist()[-1] * 1.0)
                # # print("data",data)
                # # print("$$$$$$$$$$$$$$$$audio_data: ", len(data), start,end, len(data[start:end]))
                #         soundfile.write(audio_out, data[start:end], samplerate) # mp3 save to wav
                if wav_path_callback != None:
                    wav_path_callback(audio_out)

        # 当用户停止时，直接停止函数执行
        if user_stop_flag:
            print("get_speech exit by user stop")
            return
        elif not finish_flag[0]:
            audio_generate_arg.event.clear()
            audio_generate_arg.event.wait()
        else:
            break

    print("get_speech normal exit")


def play_audio(
    audio_file="sample.wav", CHUNK=1024
):  # define play function however you like!
    print(f"~~~~~~~~~~Audio Play starts at {datetime.now()}")
    wf = wave.open(audio_file, "rb")

    p = pyaudio.PyAudio()

    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
    )

    data = wf.readframes(CHUNK)

    while len(data) != 0:
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()

    p.terminate()
    print(f"~~~~~~~~~~~~~~Audio play finishes {datetime.now()}")


def play(audio_queue):
    while 1:
        audio_out = audio_queue.get()
        if audio_out:
            if os.path.exists(audio_out):
                print("~~~playing ", audio_out)
                play_audio(audio_out)
                os.remove(audio_out)  ###############
        else:
            print("~~~~~play done")
            break


def stop_generate():
    global chat_thread, audio_thread, audio_generate_arg, user_stop_flag
    print("user stop start")
    # 设置全局用户停止标识，让各个线程快速退出
    user_stop_flag = True

    # 等待chat模型停止
    if chat_thread != None and chat_thread.is_alive():
        chat_thread.join()

    # 等待audio生成线程退出
    if audio_thread != None and audio_thread.is_alive():
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
    print("user stop finish. will clear temp resource!")
    clear_audio_cache()


def clear_audio_cache():
    global save_wav_root_path1
    try:
        # 删除文件夹及其内容
        shutil.rmtree(save_wav_root_path1)
        # 新建文件夹
        os.makedirs(save_wav_root_path1)
        print(save_wav_root_path1, "****Folder deleted and recreated.")
    except Exception as e:
        print(f"An error occurred: {e}")


import trace
import threading


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
    user_input: str,
    audio_lang=None, # 0301
    text_out_callback=None,
    llm_model_name="chatglm2-6b",
    max_token=512,
    lang="zh_CN",
    llm_first_token_callback=None,
    llm_after_token_callback=None,
    generate_audio=True,
    audio_out_callback=None,
    load_model_start_callback=None,
    load_model_finish_callback=None,
    error_callback=None,
):  ##
    global audio_thread, audio_generate_arg, chatglm2_model, tokenizer, model_path, model_name_llm, history, history_round

    # 停止上一次的生成
    stop_generate()

    top_p = 0.8
    temperature = 0.95

    print("-" * 50)
    print(
        f"user_input:{user_input} llm_model_name:{llm_model_name} max_token:{max_token} generate_audio:{generate_audio}"
    )
    ## 0305 start
    if user_input == "clear":
        clear_history()
        text_out_callback("HuatuoGPT: 你好，我是一个解答医疗健康问题的大模型，目前处于测试阶段，请以医嘱为准。请问有什么可以帮到您？输入 clear 清空对话历史")
        if llm_first_token_callback !=None:
            llm_first_token_callback("")
            
        if llm_after_token_callback !=None:
            llm_after_token_callback("")      
        return
    ## 0305 end
    
    if audio_lang == None:                         # 0301
        text_lang = langid.classify(user_input)[0] # 0301
    else:                                          # 0301
        text_lang = audio_lang                     # 0301
    print("------language: ", text_lang)

    if text_lang != "zh" and text_lang != "en" and text_lang[:2] != "ja": # 0301
        print("This program only support Chinese and English!!!!!!!!!!!!!!!")
        if text_out_callback != None:
            text_out_callback("没听清楚。请使用中文或英文复述一遍" if lang == "zh_CN" else "only Chinese and English input is supported, please change the input language and try again")
            
        if llm_first_token_callback !=None:
            llm_first_token_callback("")
            
        if llm_after_token_callback !=None:
            llm_after_token_callback("")
            
        return

    if llm_model_name == "internlm-chat-7b-8k" or llm_model_name == "internlm-chat-20b":
        prompt = INTERNLM_PROMPT_FORMAT.format(prompt=user_input)
    elif (
        llm_model_name
        == "Baichuan2-7B-Chat"
        #  or llm_model_name == "Baichuan2-13B-Chat"             ### 1205
    ):
        if text_lang[:2] == "en":  ## 0131 0301
            prompt = BAICHUAN2_PROMPT_FORMAT_en.format(prompt=user_input) ## 0301
        else:                                                            ## 0301
            prompt = BAICHUAN2_PROMPT_FORMAT.format(prompt=user_input)   ## 0301
    elif llm_model_name == "chatglm3-6b":
        if text_lang == "en":
            prompt = CHATGLM_V3_PROMPT_FORMAT_en.format(prompt=user_input)
        else:
            prompt = CHATGLM_V3_PROMPT_FORMAT.format(prompt=user_input)
    elif llm_model_name == "AquilaChat2-7B":
        prompt = AQUILA2_PROMPT_FORMAT.format(prompt=user_input)
    elif llm_model_name == "Qwen-7B-Chat":
        prompt = QWEN_PROMPT_FORMAT.format(prompt=user_input)  ##  1205
    elif llm_model_name == "HuatuoGPT-7B":                     ### 0305
        prompt = user_input #HUATUOGPT_PROMPT_FORMAT.format(prompt=user_input)  # 0305
    else:
        if text_lang == "en":
            prompt = CHATGLM_V2_PROMPT_FORMAT_en.format(prompt=user_input)
        else:
            prompt = CHATGLM_V2_PROMPT_FORMAT_zh.format(prompt=user_input)
    print("prompt: ", prompt)
    # prompt = convert(prompt, "zh-cn")

    if prompt:
        # 需要生成语音
        if generate_audio:
            print("text with audio ouput")
            text_out_finish = [False]
            audio_thread = thread_with_trace(
                target=get_speech, args=(text_out_finish, lang, audio_out_callback)
            )
            audio_thread.start()
            chat(
                prompt= prompt,
                max_length= max_token,
                top_p= top_p,
                temperature= temperature,
                llm_first_token_callback= llm_first_token_callback,
                llm_after_token_callback = llm_after_token_callback,
                llm_model_name= llm_model_name,
                text_out_callback= text_out_callback,
                generate_audio= generate_audio,
                load_model_start_callback= load_model_start_callback,
                load_model_finish_callback= load_model_finish_callback,
                error_callback=error_callback
            )
            text_out_finish[0] = True
            audio_thread.join()
        # 纯文字输出
        else:
            print("text ouput")
            chat(
                prompt= prompt,
                max_length= max_token,
                top_p= top_p,
                temperature= temperature,
                llm_first_token_callback= llm_first_token_callback,
                llm_after_token_callback=  llm_after_token_callback,
                llm_model_name= llm_model_name,
                text_out_callback= text_out_callback,
                generate_audio= generate_audio,
                load_model_start_callback= load_model_start_callback,
                load_model_finish_callback= load_model_finish_callback,
                error_callback=error_callback
            )

        print("anwser finish!")


def predict_llm_adapter(
    mode: int,
    model_name: str,
    input_data: str,
    generate_audio: bool,
    params: dict[str:any],
    lang="zh_CN",
    text_in_callback=None,
    text_out_callback=None,
    load_model_start_callback=None,
    load_model_finish_callback=None,
    sr_latency_callback=None,
    first_latency_callback=None,
    after_latency_callback=None,
    audio_out_callback=None,
    error_callback=None
):
    max_token = params.get("max_token")
    if max_token == None:
        max_token = 512
    elif not isinstance(max_token, int):
        max_token = int(max_token)
    if mode == 1:
        predict(
            audio_input=input_data,
            text_in_callback=text_in_callback,
            text_out_callback=text_out_callback,
            llm_model_name=model_name,
            max_token=max_token,
            generate_audio=generate_audio,
            lang=lang,
            audio_out_callback=audio_out_callback,
            sr_latency_callback=sr_latency_callback,
            llm_first_token_callback=first_latency_callback,
            llm_after_token_callback=after_latency_callback,
            load_model_start_callback=load_model_start_callback,
            load_model_finish_callback=load_model_finish_callback,
            error_callback= error_callback
        )
    else:
        if sr_latency_callback != None:
            sr_latency_callback("")
        predict_user(
            user_input=input_data,
            audio_lang=None, # 0301
            generate_audio=generate_audio,
            llm_model_name=model_name,
            max_token=max_token,
            lang=lang,
            text_out_callback=text_out_callback,
            audio_out_callback=audio_out_callback,
            load_model_start_callback=load_model_start_callback,
            load_model_finish_callback=load_model_finish_callback,
            llm_first_token_callback=first_latency_callback,
            llm_after_token_callback=after_latency_callback,
            error_callback = error_callback
        )


def predict(
    audio_input,
    text_in_callback=None,
    text_out_callback=None,
    llm_model_name="chatglm2-6b",
    max_token=512,
    lang="zh_CN",
    llm_first_token_callback=None,
    llm_after_token_callback=None,
    sr_latency_callback=None,
    generate_audio=True,
    audio_out_callback=None,
    load_model_start_callback=None,
    load_model_finish_callback=None,
    error_callback=None
):  ## miss step
    
    clear_audio_cache()
    global processor, whisper, device_select_sr,asr_model, ct_model

    print("audio to text start")
    torch.xpu.synchronize()

    t0 = time.time()
    print("save wav  -----", audio_input)
   # prompt_in = get_prompt(processor, whisper, audio_input, device_select_sr)
    prompt_in, bef_prompt_in = get_prompt_funasr(asr_model, ct_model, audio_input)   ## 0124 0301
    prompt_in = convert(prompt_in, "zh-cn")
    print("\n ******** 2")
    torch.xpu.synchronize()
    t1 = time.time()
    sr_latency_count = (t1 - t0) * 1000

    print("sr_latency(ms): ", sr_latency_count)
    if sr_latency_callback != None:
        sr_latency_callback(str(round(sr_latency_count, 2)) + " ms")

    audio_lang = langid.classify(bef_prompt_in)[0]  ## 0301
    print("------language: ", audio_lang)       ## 1206
    print("------prompt_in: ", prompt_in)       ## 0131

    if audio_lang != "zh" and audio_lang != "en" and audio_lang[:2] != "ja":     ## 1206
        print("This program only support Chinese and English!!!!!!!!!!!!!!!")  ## 1206

        if text_in_callback != None:
            text_in_callback("未能识别音频" if lang == "zh_CN" else "Failure to recognize audio")

        if text_out_callback != None:
            text_out_callback("对不起，没听清楚。请使用中文或英文复述一遍" if lang == "zh_CN" else "sorry, I can't hear you. Please use Chinese or English Say it again")
            
        if llm_first_token_callback !=None:
            llm_first_token_callback("")
            
        if llm_after_token_callback !=None:
            llm_after_token_callback("")
            
        return
    else:
        if text_in_callback != None:
            text_in_callback(prompt_in)

    predict_user(
        user_input= prompt_in,
        audio_lang=audio_lang, # 0301
        text_out_callback= text_out_callback,
        llm_model_name= llm_model_name,
        max_token = max_token,
        llm_first_token_callback= llm_first_token_callback,
        llm_after_token_callback= llm_after_token_callback,
       generate_audio= generate_audio,
        audio_out_callback= audio_out_callback,
        load_model_start_callback= load_model_start_callback,
        load_model_finish_callback= load_model_finish_callback,
        error_callback=error_callback
    )


def dispose():
    global chatglm2_model
    stop_generate()
    if chatglm2_model != None:
        try:
            chatglm2_model.to("cpu")
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
            del chatglm2_model
            gc.collect()
        except:
            None


def dispose_asr_tts(): ## 0129
    global asr_model, ct_model,tts_executor
  #  stop_generate()
    if asr_model != None:
        try:
            del asr_model, ct_model
            gc.collect()
        except:
            None
    if tts_executor != None:
        try:
            del tts_executor
            gc.collect()
        except:
            None

def choose_function(user_function):
    if user_function == "语音助手":
        return [
            gr.update(value="", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ]
        # return [gr.Chatbot(label="", scale=1, avatar_images=["chatGPT.png", "chatGPT.png"], visible=True),
        #        gr.Image(scale=1,label="Image for txt2img", show_label=False, type="pil", tool="editor", image_mode="RGBA", height=540, visible=False)]
    else:
        # return [gr.Chatbot(label="", scale=1, avatar_images=["chatGPT.png", "chatGPT.png"], visible=False),
        #        gr.Image(scale=1,label="Image for txt2img", show_label=False, type="pil", tool="editor", image_mode="RGBA", height=540, visible=True)]
        return [
            gr.update(value="", visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
        ]


def reset_user_input():
    return gr.update(value="")


def reset_state():
    return [], [], gr.update(value=""), gr.update(value=""), gr.update(value="")

def get_init_settings_bak():
    global debug_mode, model_path

    model_list = list()
    
    filenames = ["aigc_setting_debug.json","aigc_setting.json"] if debug_mode else ["aigc_setting.json"]
    for name in os.listdir(model_path):
        # 在DEBUG模式下优先加载sd_cfg_debug.json，不存在时继续加载sd_cfg_.json
        for filename in filenames:
            cfg_path = os.path.join(model_path,name, filename)
            if os.path.exists(cfg_path):
                model_list.append(load_json_from_file(cfg_path))
                break
    
    return { "modelList": model_list}

def get_init_settings():
    global debug_mode, model_path

    model_list = [
        # {
        #     "model": "Baichuan2-7B-Chat",
        #     "params": [
        #         {
        #             "name": "inference",
        #             "languages": {
        #                 "zh_CN": "推理设备",
        #                 "en_US": "inference deveice",
        #             },
        #             "type": "display",
        #             "value": "iGPU",
        #         },
        #         {
        #             "name": "max_token",
        #             "languages": {"zh_CN": "最大长度", "en_US": "max token"},
        #             "type": "slidebar",
        #             "min": 1,
        #             "max": 1024,
        #             "step": 32,
        #             "value": 512,
        #         },
                
        #     ],
        # },
        {
            "model": "HuatuoGPT-7B",
            "params": [
                {
                    "name": "inference",
                    "languages": {"zh_CN": "推理设备", "en_US": "inference deveice"},
                    "type": "display",
                    "value": "iGPU",
                },
                {
                    "name": "max_token",
                    "languages": {"zh_CN": "最大长度", "en_US": "max token"},
                    "type": "slidebar",
                    "min": 1,
                    "max": 1024,
                    "step": 32,
                    "value": 512,
                },
            ],
        },
    ]
        

    exists_models = set()
    for name in os.listdir(model_path):
        exists_models.add(name.replace("-int4",""))

    print("debug_mode", debug_mode)
    exists_model_list = []
    for item in model_list:
        if item.get("model") in exists_models:
            if debug_mode:
                params = item.get("params")
                params.append({
                    "name": "sr_latency",
                    "languages": {"zh_CN": "语音识别耗时", "en_US": "sr latency"},
                    "type": "display",
                    "value": "",
                })
                params.append({
                    "name": "first_token_latency",
                    "languages": {
                        "zh_CN": "首字生成耗时",
                        "en_US": "first token latency",
                    },
                    "type": "display",
                    "value": "",
                })
                params.append({
                    "name": "after_token_latency",
                    "languages": {
                        "zh_CN": "平均吐字耗时",
                        "en_US": "after token latency",
                    },
                    "type": "display",
                    "value": "",
                })
            exists_model_list.append(item)
    
    return { "modelList": exists_model_list}
            

# https://github.com/THUDM/ChatGLM2-6B/blob/main/web_demo2.py
# if __name__ == "__main__":
def model_list_f():
    model_list_all = list[str]()
    for model in get_init_settings().get("modelList"):
        model_list_all.append(model.get("model"))
    return model_list_all


def init(debug=False, llm_init=None, save_wav_root_path="./static/audio_cache"):
    # global model_path, device, model_load, whisper, processor, chatglm2_model, tokenizer, model_loaded, tts_model, tts_en_model, pipe, frontend, am_sess, voc_sess, spk_id, t1, t2, t3
    global debug_mode, model_path, device, model_load, whisper, processor, chatglm2_model, tokenizer, model_loaded, pipe, tts_executor, chat_thread, audio_thread, user_stop_flag, audio_generate_arg, sr_model_now, device_sr, model_name_llm, device_select_sr, save_wav_root_path1,asr_model, ct_model,max_len_history # 0305  # tts_frontend, am_sess, voc_sess, spk_id,tts_en_model,
    debug_mode = debug
    model_path = "./models/llm/"
    model_list_all = model_list_f()
    print("model list: ", model_list_all)
    device_list = ["iGPU", "CPU"]
    # device_list = ["iGPU"]
    device = "None"
    device_sr = "iGPU"
    model_load = False
    chat_thread = None
    audio_thread = None
    user_stop_flag = False
    audio_generate_arg = AudioGenerateArg()
    sr_model_now = "whisper-small"
    max_len_history = 0

    if llm_init == None:
        # model_name_llm = model_list_all[0]  #### 1205
        model_name_llm = None  #### 1205
    else:
        if llm_init in model_list_all:
            model_name_llm = llm_init
        else:
            return "Error: wrong model name or local device don't have this model."

    save_wav_root_path1 = save_wav_root_path

    device_select = "xpu"  # if device_name == "dGPU" else  "cpu"
    device_select_sr = "xpu"
    model_loaded = False

    #processor, whisper = load_whisper_model(model_path, sr_model_now, device_select_sr)

    #asr_model, ct_model = load_asr_model(model_path)  ## 0124

    print("loading tts fastspeech2_mix paddle ---------")
    tts_executor = TTSExecutor()
    # whisper = load_whisper_model_cpu(model_path, "cpu")  ### 1205
    # chatglm2_model, tokenizer = load_chatglm2_model(
    #    model_path, model_name_llm, device_select
    # )
    chatglm2_model, tokenizer = None, None  #### 1205
    # tts_model, tts_en_model = load_tts_model(model_path, device_select)

    if not os.path.exists(save_wav_root_path1):
        os.makedirs(save_wav_root_path1)

    load_tts_model_paddle("llm service start")


from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    STOPPING_CRITERIA_INPUTS_DOCSTRING,
    add_start_docstrings,
)


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

    def __init__(self):
        """
        音频生成控制参数对象
        """
        self.segment_queue = Queue(-1)
        self.event = threading.Event()
