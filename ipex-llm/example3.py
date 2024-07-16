import torch
#import intel_extension_for_pytorch as ipex
#from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel
from ipex_llm.transformers import AutoModelForCausalLM, AutoModel
from transformers import AutoTokenizer

import time
import numpy as np
#from benchmark_util import BenchmarkWrapper
from transformers import TextIteratorStreamer
def stream_chat(model, tokenizer, prompt, max_new_tokens, history=[], device="cpu"):
    # format conversation context as prompt through chat history
    #prompt = CHATGLM_V2_PROMPT_FORMAT.format(prompt=input_str)
    #prompt = LLAMA2_PROMPT_FORMAT.format(prompt=input)
    input_ids = tokenizer([prompt], return_tensors='pt').to(device)
    print("stream_chat-----input_ids:", input_ids)

    streamer = TextIteratorStreamer(tokenizer,
                                    skip_prompt=True, # skip prompt in the generated tokens
                                    skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids,
        streamer=streamer,
        num_beams=1,
        do_sample=False,
        max_new_tokens=max_new_tokens
    )

    # to ensure non-blocking access to the generated text, generation process should be ran in a separate thread
    from threading import Thread

    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()
    history = []

    output_str = ""
    for stream_output in streamer:
        output_str += stream_output
        yield output_str, history


model_path = r"C:\\Program Files\\AIGC\\resources\\service\\models\\llm\\Qwen2-1.5B-int4"

prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

#prompt = "In the year 2048, the world was a very different place from what it had been just two decades before. The pace of technological progress had quickened to an almost unimaginable degree, and the changes that had swept through society as a result were nothing short of revolutionary. In many ways, the year 2048 represented the culmination of a long and tumultuous journey that humanity had been on since the dawn of civilization. The great leaps forward in science and technology that had occurred over the course of the previous century had laid the groundwork for a future that was beyond anything anyone could have imagined. One of the most striking aspects of life in 2048 was the degree to which technology had become an integral part of nearly every aspect of daily existence. From the moment people woke up in the morning until they went to bed at night, they were surrounded by devices and systems that were powered by advanced artificial intelligence and machine learning algorithms. In fact, it was hard to find anything in people's lives that wasn't touched by technology in some way. Every aspect of society had been transformed, from the way people communicated with one another to the way they worked, played, and even socialized. And as the years went on, it seemed as though there was no limit to what technology could achieve. Despite all of these advances, however, not everyone was happy with the state of the world in 2048. Some people saw the increasing reliance on technology as a sign that humanity was losing touch with its own humanity, and they worried about the implications of this for the future. Others were more pragmatic, recognizing that while technology had brought many benefits, it also posed new challenges and risks that needed to be addressed. As a result, there was a growing movement of people who were working to ensure that the advances of technology were used in ways that were safe, ethical, and beneficial for everyone. One person who was at the forefront of this movement was a young woman named Maya. Maya was a brilliant and ambitious researcher who had dedicated her life to understanding the implications of emerging technologies like artificial intelligence and biotechnology. She was deeply concerned about the potential risks and unintended consequences of these technologies, and she worked tirelessly to raise awareness about the need for responsible innovation. Maya's work had earned her a reputation as one of the most influential voices in the field of technology and ethics, and she was widely respected for her deep understanding of the issues and her ability to communicate complex ideas in ways that were accessible and engaging. She was also known for her passionate and inspiring speeches, which often left her audiences with a sense of purpose and determination to make the world a better place through their own efforts. One day, Maya received an invitation to speak at a major conference on technology and ethics, which was being held in a large convention center in the heart of the city. The conference was expected to attract thousands of people from all over the world, and there was a great deal of excitement and anticipation about what Maya would say. As she prepared for her speech, Maya knew that she had a big responsibility on her shoulders. She felt a deep sense of obligation to use her platform to inspire others to take action and make a difference in the world, and she was determined to do everything in her power to live up to this responsibility. When the day of the conference arrived, Maya was filled with a mixture of excitement and nerves. She spent hours rehearsing her speech and fine-tuning her ideas, making sure that she had everything just right. Finally, after what felt like an eternity, it was time for her to take the stage. As she stepped up to the podium, Maya could feel the energy of the crowd surging around her. She took a deep breath and began to speak, her voice strong and clear as she outlined the challenges and opportunities facing society in the age of technology. She spoke passionately about the need for responsible innovation and the importance of considering the ethical implications of our actions, and she inspired many people in the audience to take up this cause and make a difference in their own lives. Overall, Maya's speech was a resounding success, and she received countless messages of gratitude and appreciation from those who had heard her speak. She knew that there was still much work to be done, but she felt hopeful about the future and the role that technology could play in creating a better world for all. As Maya left the stage and made her way back to her seat, she couldn't help but feel a sense of pride and accomplishment at what she had just accomplished. She knew that her words had the power to inspire others and make a real difference in the world, and she was grateful for the opportunity to have played a part in this important work. For Maya, the future was full of promise and possibility, and she was determined to continue doing everything in her power to help create a brighter, more ethical world for everyone. As she "

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

#model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True,use_cache=True,low_cpu_mem_usage=True,cpu_embedding=True).eval()
model = AutoModelForCausalLM.load_low_bit(model_path, trust_remote_code=True, cpu_embedding=True).eval()

#model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,low_cpu_mem_usage=True, torch_dtype=torch.half).eval()
#model.rwkv._rescale_layers()

#model = AutoModel.from_pretrained(model_path, trust_remote_code=True, optimize_model=False, load_in_4bit=True).eval()
#model = AutoModel.load_low_bit(model_path, trust_remote_code=True, optimize_model=True).eval()

if 0:
    model.save_low_bit(model_path +  "-int4/")
    tokenizer.save_pretrained(model_path + "-int4/")

#model = AutoModelForCausalLM.load_low_bit(model_path + "-int4", trust_remote_code=True, optimize_model=True).eval()


input_ids = tokenizer.encode(prompt, return_tensors="pt")
print("finish to load")

model = model.to('cpu')
#model.model.embed_tokens.to('cpu')
#model.transformer.embedding.to('cpu')
input_ids = input_ids.to('cpu')


print("finish to xpu")


for _ in range(5):
    response_ = ""
    response = ""
    timeFirst = 0
    timeFirstRecord = False
    torch.xpu.synchronize()
    timeStart = time.time()
   # prompt = CHATGLM_V2_PROMPT_FORMAT.format(prompt=prompt)
    with torch.inference_mode():
        for response, history in stream_chat(model, tokenizer, prompt,32):
           # chatbot[-1] = (input, parse_text(response))
            print(response.replace(response_, ""), end="")  ###  !!!! return 
            response_ = response
            if timeFirstRecord == False:
                torch.xpu.synchronize()
                timeFirst = time.time() - timeStart
                timeFirstRecord = True
           # yield chatbot, history,  "", ""
        timeCost = time.time() - timeStart
    token_count_input = len(tokenizer.tokenize(prompt))
    token_count_output = len(tokenizer.tokenize(response))
    ms_first_token = timeFirst * 1000
    ms_after_token = (timeCost - timeFirst) / (token_count_output - 1+1e-8) * 1000
    print("input: ", prompt)
    print("output: ", response)
    print("token count input: ", token_count_input)
    print("token count output: ", token_count_output)
    print("time cost(s): ", timeCost)
    print("First token latency(ms): ", ms_first_token)
    print("After token latency(ms/token)", ms_after_token)