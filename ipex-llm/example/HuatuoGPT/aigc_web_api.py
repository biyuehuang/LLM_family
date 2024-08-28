import json
import signal
import traceback
from flask import Flask, jsonify, request, Response, stream_with_context
import os
import time
import web_utils
from flask_cors import CORS
import LLM_mtl_v02
import audiosd_lcm_v07
import llm_web_adapter
import sd_web_adapter
import sys
import audioRecorder
import audioToText

B64_HEAD_MP3 = "data:audio/mpeg;base64,"
B64_HEAD_WAV = "data:audio/wav;base64,"

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["POST,GET"]}})
recorder: audioRecorder.Recorder

debug_mode : bool

benchmark_support = os.path.exists("./AIbenchmark")

if benchmark_support:
    sys.path.append(os.path.abspath("./AIbenchmark"))

@app.after_request
def after(resp: Response):
    """
    被after_request钩子函数装饰过的视图函数
    ，会在请求得到响应后返回给用户前调用，也就是说，这个时候，
    请求已经被app.route装饰的函数响应过了，已经形成了response，这个时
    候我们可以对response进行一些列操作，我们在这个钩子函数中添加headers，所有的url跨域请求都会允许！！！
    """
    resp.headers.set("Access-Control-Allow-Origin", request.headers.get("Origin"))
    resp.headers.set("Access-Control-Allow-Methods", "GET,POST")
    resp.headers.set("Access-Control-Allow-Headers", "x-requested-with,content-type")
    return resp


@app.route("/api/status", methods=["GET"])
def status():
    return "ok"

# LLM初始化
@app.route("/api/llm/init", methods=["GET"])
def llm_init():
   # if audiosd_lcm_v07.inited == 2:
   #     audiosd_lcm_v07.dispose()
        
   # if LLM_mtl_v02.inited == 0:
    if 1:
        LLM_mtl_v02.init(
            debug=debug_mode,
            save_wav_root_path=os.path.join(os.getcwd(), "static/audio_cache"),
        )
    return jsonify(LLM_mtl_v02.get_init_settings())

# LLM停止生成
@app.route("/api/llm/stopGenerate", methods=["GET"])
def llm_stop_generate():
    return jsonify(LLM_mtl_v02.stop_generate())


# LLM聊天生成
@app.route("/api/llm/chat", methods=["POST"])
def llm_chat():
    headers = {
        "Access-Control-Allow-Origin": request.headers.get("Origin"),
        "Access-Control-Allow-Methods": "GET,POST",
        "Access-Control-Allow-Headers": "x-requested-with,content-type",
    }
    lang = request.form.get("lang")
    input_type = int(request.form.get("inputType"))
    model = request.form.get("model")
    enable_audio_opt = int(request.form.get("enableAudioOpt")) == 1
    params = json.loads(request.form.get("params"))
    sse_invoker = llm_web_adapter.LLM_SSE_Invoker(request.url_root)
    # 语音输入
    if input_type == 2:
        input_data = request.form.get("inputData")
        it = sse_invoker.audio_conversation(
            wave_file=input_data,
            model=model,
            params=params,
            generate_audio=enable_audio_opt,
            ui_lang=lang,
        )
        # 返回Response对象，使用stream_with_context来创建事件流
        return Response(
            stream_with_context(it), content_type="text/event-stream", headers=headers
        )
    else:
        input_data = request.form.get("inputData")
        it = sse_invoker.text_conversation(
            input_data=input_data,
            model=model,
            params=params,
            generate_audio=enable_audio_opt,
            lang=lang,
        )
        # 返回Response对象，使用stream_with_context来创建事件流
        return Response(
            stream_with_context(it), content_type="text/event-stream", headers=headers
        )

# 清空LLM聊天记录
@app.route("/api/llm/clearHistory", methods=["GET"])
def llm_clear_history():
    #if LLM_mtl_v02.inited:
    LLM_mtl_v02.clear_history()

# SD初始化
@app.route("/api/sd/init", methods=["GET"])
def init_sd():
    if LLM_mtl_v02.inited == 2:
        LLM_mtl_v02.dispose()
        
    if audiosd_lcm_v07.inited == 0:
        audiosd_lcm_v07.init(
            debug=debug_mode,
            image_save_dir=os.path.join(os.getcwd(), "static/images"),
            model_find_path=os.path.join(os.getcwd(), "models/sd"))
    
    return jsonify(audiosd_lcm_v07.get_init_settings())

@app.route("/api/sd/textToPaint", methods=["POST"])
def text_to_paint():
    headers = {
        "Access-Control-Allow-Origin": request.headers.get("Origin"),
        "Access-Control-Allow-Methods": "GET,POST",
        "Access-Control-Allow-Headers": "x-requested-with,content-type",
    }
    input_type = int(request.form["inputType"])
    model_name = request.form["model"]
    params_json = request.form["params"]
    params = json.loads(params_json)
    text_input: str = None
    audio_input: str = None
    if input_type == 2:
        audio_input = request.form.get("inputData")
    else:
        text_input = request.form["inputData"]
    sse_invoker = sd_web_adapter.SD_SSE_Invoker(request.url_root)
    it = sse_invoker.generate_image(
        model_name=model_name,
        mode=0,
        params=params,
        text_input=text_input,
        audio_input=audio_input,
    )
    return Response(
        stream_with_context(it), content_type="text/event-stream", headers=headers
    )


@app.route("/api/sd/imageToPaint", methods=["POST"])
def image_to_paint():
    headers = {
        "Access-Control-Allow-Origin": request.headers.get("Origin"),
        "Access-Control-Allow-Methods": "GET,POST",
        "Access-Control-Allow-Headers": "x-requested-with,content-type",
    }
    src_image = request.files["srcImage"]
    src_image_path = os.path.join(
        web_utils.get_temp_folder(),
        f"{time.time()}{web_utils.get_file_extension(src_image.filename)}"
    )
    src_image.save(src_image_path)

    input_type = int(request.form["inputType"])
    model_name = request.form["model"]
    params = json.loads(request.form["params"])
    text_input: str = None
    audio_input: str = None
    if input_type == 2:
        audio_input = request.form.get("inputData")
    else:
        text_input = request.form["inputData"]
    sse_invoker = sd_web_adapter.SD_SSE_Invoker(request.url_root)
    it = sse_invoker.generate_image(
        model_name=model_name,
        mode=1,
        params=params,
        text_input=text_input,
        audio_input=audio_input,
        input_image=src_image_path,
    )
    return Response(
        stream_with_context(it), content_type="text/event-stream", headers=headers
    )


@app.route("/api/sd/controlnetPaint", methods=["POST"])
def controlnet_paint():
    src_image = request.files["srcImage"]
    src_image_path = os.path.join(
        web_utils.get_temp_folder(),
        f"{time.time()}{web_utils.get_file_extension(src_image.filename)}"
    )
    src_image.save(src_image_path)
    input_type = int(request.form["inputType"])
    model_name = request.form["model"]
    params = json.loads(request.form["params"])
    input_type = int(request.form["inputType"])
    text_input: str = None
    audio_input: str = None
    if input_type == 2:
        audio_input = request.form.get("inputData")
    else:
        text_input = request.form["inputData"]
    headers = {
        "Access-Control-Allow-Origin": request.headers.get("Origin"),
        "Access-Control-Allow-Methods": "GET,POST",
        "Access-Control-Allow-Headers": "x-requested-with,content-type",
    }
    sse_invoker = sd_web_adapter.SD_SSE_Invoker(request.url_root)
    it = sse_invoker.generate_image(
        model_name=model_name,
        mode=2,
        params=params,
        text_input=text_input,
        audio_input=audio_input,
        input_image=src_image_path,
    )
    return Response(
        stream_with_context(it), content_type="text/event-stream", headers=headers
    )


@app.route("/api/sd/inpaint", methods=["POST"])
def inpaint():
    src_image = request.files["srcImage"]
    src_image_path = os.path.join(
        web_utils.get_temp_folder(),
        f"{time.time()}{web_utils.get_file_extension(src_image.filename)}"
    )
    src_image.save(src_image_path)
    modify_mask = request.files["modifyMask"]
    mask_width = int(request.form["maskWidth"])
    mask_height = int(request.form["maskHeight"])
    rgb_mask = web_utils.generate_mask_image(
        modify_mask.stream.read(), mask_width, mask_height
    )
    mask_image_path = os.path.join(web_utils.get_temp_folder(), f"{time.time()}.jpg")
    rgb_mask.save(mask_image_path)
    input_type = int(request.form["inputType"])
    model_name = request.form["model"]
    params = json.loads(request.form["params"])
    input_type = int(request.form["inputType"])
    text_input: str = None
    audio_input: str = None
    if input_type == 2:
        audio_input = request.form.get("inputData")
    else:
        text_input = request.form["inputData"]
    headers = {
        "Access-Control-Allow-Origin": request.headers.get("Origin"),
        "Access-Control-Allow-Methods": "GET,POST",
        "Access-Control-Allow-Headers": "x-requested-with,content-type",
    }
    sse_invoker = sd_web_adapter.SD_SSE_Invoker(request.url_root)
    it = sse_invoker.generate_image(
        model_name=model_name,
        mode=3,
        params=params,
        text_input=text_input,
        audio_input=audio_input,
        input_image=src_image_path,
        input_mask=mask_image_path,
    )
    return Response(
        stream_with_context(it), content_type="text/event-stream", headers=headers
    )

# @app.route("/api/sd/upsample", methods=["POST"])
# def upsample():
#     src_image = request.files["srcImage"]
#     src_image_path = os.path.join(
#         web_utils.get_temp_folder(),
#         f"{time.time()}{web_utils.get_file_extension(src_image.filename)}"
#     )
#     generate_path = audiosd_lcm_v07.upsample(src_image_path)
#     os.remove(src_image_path)
#     return jsonify({"code": 1, "message": "success", "url":  f"{request.url_root}/static/images/{os.path.basename(generate_path)}" })

@app.route("/api/sd/upsample", methods=["POST"])
def upsample():
    src_image = request.files["srcImage"]
    image_bytes = src_image.stream.read()
    src_image.close()
    generate_path = audiosd_lcm_v07.upsample(image_bytes)
    if audiosd_lcm_v07.set_windows_wallpaper(generate_path):
        return jsonify({"code": 1, "message": "success"})
    else:
        return jsonify({"code": 0, "message": "set wallpaper faild"})


@app.route("/api/applicationExit", methods=["GET"])
def applicationExit():
    try:
        if LLM_mtl_v02.inited:
            LLM_mtl_v02.dispose()
        if audiosd_lcm_v07.inited:
            audiosd_lcm_v07.dispose()
    finally:
        pid = os.getpid()
        os.kill(pid, signal.SIGINT)


@app.route("/api/startRecordAudio", methods=["GET"])
def startRecordAudio():
    recorder.start_record()
    return jsonify({"code": 1, "message": "success"})


@app.route("/api/stopRecordAudio", methods=["POST"])
def stopRecordAudio():
    try:
        cancel = request.form.get("cacncel")
        wave_file = recorder.stop_record(cancel == '1')
        return jsonify({"code": 1, "message": "success", "wave_file": wave_file})
    except:
        traceback.print_exc()
        return jsonify({"code": 0, "message": "failed"})

@app.route("/api/startBenchmark")
def startBenchmark():
    if not benchmark_support:
        return "no surrpot"
    
    import benchmark_web_adapter
    audioToText.dispose()
    try:
        if LLM_mtl_v02.inited:
            LLM_mtl_v02.dispose()
        if audiosd_lcm_v07.inited:
            audiosd_lcm_v07.dispose()
    except:
        None
    headers = {
        "Access-Control-Allow-Origin": request.headers.get("Origin"),
        "Access-Control-Allow-Methods": "GET,POST",
        "Access-Control-Allow-Headers": "x-requested-with,content-type",
    }
    adapter = benchmark_web_adapter.Benchmark_Web_Adapter()
    it = adapter.start(os.getcwd())
    return Response(
        stream_with_context(it), content_type="text/event-stream", headers=headers
    )

@app.route("/api/stopBenchmark")
def stopBenchmark():
    if not benchmark_support:
        return "no surrpot"
    
    import benchmark
    benchmark.stop()
    return "ok"

if __name__ == "__main__":
    recorder = audioRecorder.Recorder(os.path.join(os.getcwd(), "temp"))
    audioToText.init(os.path.join(os.getcwd(), "models/common"))
    debug_mode = sys.argv.__len__() > 1 and sys.argv[1] == "--debug"
    app.run(host="127.0.0.1", port=34567)
