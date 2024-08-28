import threading
from queue import Empty, Queue
import json
import os
import time
import LLM_mtl_v02
import traceback
import customException


class LLM_SSE_Invoker:
    msg_queue: Queue
    finish: bool
    singal: threading.Event
    url_root: str

    def __init__(self, url_root):
        self.msg_queue = Queue(-1)
        self.finish = False
        self.singal = threading.Event()
        self.url_root = url_root

    def put_msg(self, data):
        self.msg_queue.put_nowait(data)
        self.singal.set()

    def text_in_callback(self, msg: str):
        data = {"type": "text_in", "value": msg}
        self.put_msg(data)

    def text_out_callback(self, msg: str):
        data = {"type": "text_out", "value": msg}
        self.put_msg(data)
    

    def audio_out_callback(self, file_path: str):
        if os.path.exists(file_path):
            data = {
                "type": "audio_out",
                "value": f"{self.url_root}/static/audio_cache/{os.path.basename(file_path)}",
            }
            self.put_msg(data)

    def first_latency_callback(self, first_latency: str):
        data = {"type": "first_token_latency", "value": first_latency}
        self.put_msg(data)

    def after_latency_callback(self, after_latency: str):
        data = {"type": "after_token_latency", "value": after_latency}
        self.put_msg(data)

    def sr_latency_callback(self, sr_latency: str):
        data = {"type": "sr_latency", "value": sr_latency}
        self.put_msg(data)

    def load_model_start_callback(self, model_name: str):
        if model_name != None:
            data = {"type": "load_model_start", "value": f"{model_name} ms"}
            self.put_msg(data)

    def load_model_finish_callback(self, model_name: str):
        if model_name != None:
            data = {"type": "load_model_finish", "value": f"{model_name} ms"}
            self.put_msg(data)
    
    def error_callback(self, ex: Exception):
        print(f"exception:{str(ex)}")
        if isinstance(ex, RuntimeError):
            self.put_msg( {"type": "error", "value": "RuntimeError"})
        elif isinstance(ex, customException.SpeechRecognitionException):
            self.put_msg( {"type": "error", "value": "SpeechRecognitionException"})
        elif isinstance(ex, customException.SensitiveWordsException):
            self.put_msg( {"type": "error", "value": "SensitiveWordsException"})
        else:
            self.put_msg( {"type": "error", "value": "Exception"})
            
    def audio_conversation(
        self, wave_file: str, model: str, params: dict[str:any], generate_audio: bool, ui_lang: str
    ):
        thread = threading.Thread(
            target=self.audio_conversation_run,
            args=(wave_file, model, params, generate_audio,ui_lang),
        )
        thread.start()
        return self.generator()

    def audio_conversation_run(
        self,
        wave_file: str,
        model_name: str,
        params: dict[str:any],
        generate_audio: bool,
        ui_lang:str
    ):
        try:
            LLM_mtl_v02.predict_llm_adapter(
                mode=1,
                model_name=model_name,
                input_data=wave_file,
                generate_audio=generate_audio,
                ui_lang=ui_lang,
                params=params,
                text_in_callback=self.text_in_callback,
                text_out_callback=self.text_out_callback,
                load_model_start_callback=self.load_model_start_callback,
                load_model_finish_callback=self.load_model_finish_callback,
                sr_latency_callback=self.sr_latency_callback,
                first_latency_callback=self.first_latency_callback,
                after_latency_callback=self.after_latency_callback,
                audio_out_callback=self.audio_out_callback,
                error_callback = self.error_callback
            )
            self.put_msg( {"type": "finish"})
        except Exception as ex:
            traceback.print_exc()
            self.error_callback(ex)
        finally:
            os.remove(wave_file)
            self.finish = True
            self.singal.set()

    def text_conversation(
        self, input_data: str, model: str, params: dict[str:any], generate_audio: bool, lang: str
    ):
        thread = threading.Thread(
            target=self.text_conversation_run,
            args=(input_data, model, params, generate_audio,lang),
        )
        thread.start()
        return self.generator()

    def text_conversation_run(
        self,
        input_text: str,
        model_name: str,
        params: dict[str:any],
        generate_audio: bool,
        lang:str
    ):
        try:
            LLM_mtl_v02.predict_llm_adapter(
                mode=0,
                model_name=model_name,
                input_data=input_text,
                generate_audio=generate_audio,
                params=params,
                lang=lang,
                text_in_callback=self.text_in_callback,
                text_out_callback=self.text_out_callback,
                load_model_start_callback=self.load_model_start_callback,
                load_model_finish_callback=self.load_model_finish_callback,
                sr_latency_callback=self.sr_latency_callback,
                first_latency_callback=self.first_latency_callback,
                after_latency_callback=self.after_latency_callback,
                audio_out_callback=self.audio_out_callback,
                error_callback = self.error_callback,
            )
            self.put_msg({"type": "finish"})
        except Exception as ex:
            traceback.print_exc()
            self.error_callback(ex)
        finally:
            self.finish = True
            self.singal.set()

    def generator(self):
        while True:
            while not self.msg_queue.empty():
                try:
                    data = self.msg_queue.get_nowait()
                    msg = f"data:{json.dumps(data)}\0"
                    yield msg
                except Empty(Exception):
                    break
            if not self.finish:
                self.singal.clear()
                self.singal.wait()
            else:
                break
