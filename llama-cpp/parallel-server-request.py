# llama-server.exe -m "C:\Users\user\Desktop\model\gpt-oss-20b-Q4_K_M.gguf" -e -s 0 --host 127.0.0.1 --port 8080 -n 50 -mg 0 -ngl 100 --parallel 1 --no-context-shift --slot-save-path "C:\Users\user\slot" -fa on --swa-full -c 8292 --ignore-eos
"""
Parallel server request test script.
    usage: python3 parallel-server-request.py 1 1024.txt
"""

import json
import time
import sys
import random
from multiprocessing import Process
from pathlib import Path
from typing import Final

import requests

LLM_SERVER_URL: Final[str] = "http://127.0.0.1:8080"
NUM_INSTANCES: Final[int] = int(sys.argv[1]) if len(sys.argv) > 1 else 1
file_names: Final[str] = str(sys.argv[2])
CONNECT_TIMEOUT: float = 300.0
RESPONSE_TIMEOUT: float = 300.0
MAXIMUM_RUNTIME: int = 3000
MAX_TOKENS: int = 512

def clean_text(text):
    """清理文本：去除换行符、所有引号"""
    # 替换换行符、回车符为空
    cleaned = text.replace('\n', '').replace('\r', '')
    # 替换单引号、双引号为空
    cleaned = cleaned.replace('"', '').replace("'", '')
    return cleaned

file_names = [file_names]
# 定义要读取的所有文件名
#file_names = [
  #  "32.txt",
  #  "1024.txt",
  #  "2k.txt",
  #  "4k.txt",
  #  "8k.txt",
  #  "16k.txt",
  #  "32k.txt"
#]

# 存储所有处理后的内容
PROMPTS = []

# 循环读取每个文件
for file in file_names:
    try:
        # 读取文件内容（使用 utf-8 编码，兼容中文）
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 清理内容：无换行、无引号
        clean_content = clean_text(content)
        
        # 添加到 prompt 列表
        PROMPTS.append(clean_content)
        print(f"✅ 已读取并处理：{file}")
    
    except FileNotFoundError:
        print(f"❌ 未找到文件：{file}")
    except Exception as e:
        print(f"⚠️ 处理 {file} 时出错：{str(e)}")


print("\n最终生成的 prompt 列表：")

####################################

#PROMPTS = ["In the year 2048, the world was a very different place from what it had been just two decades before. The pace of technological progress had quickened to an # almost unimaginable degree, and the changes that had swept through society as a result were nothing short of revolutionary. In many ways, the year 2048 represented the culmination of a long and tumultuous journey that humanity had been on since the dawn of civilization. The great leaps forward in science and technology that had occurred over the course of the previous century had laid the groundwork for a future that was beyond anything anyone could have imagined. One of the most striking aspects of life in 2048 was the degree to which technology had become an integral part of nearly every aspect of daily existence. From the moment people woke up in the morning until they went to bed at night, they were surrounded by devices and systems that were powered by advanced artificial intelligence and machine learning algorithms. In fact, it was hard to find anything in people's lives that wasn't touched by technology in some way. Every aspect of society had been transformed, from the way people communicated with one another to the way they worked, played, and even socialized. And as the years went on, it seemed as though there was no limit to what technology could achieve. Despite all of these advances, however, not everyone was happy with the state of the world in 2048. Some people saw the increasing reliance on technology as a sign that humanity was losing touch with its own humanity, and they worried about the implications of this for the future. Others were more pragmatic, recognizing that while technology had brought many benefits, it also posed new challenges and risks that needed to be addressed. As a result, there was a growing movement of people who were working to ensure that the advances of technology were used in ways that were safe, ethical, and beneficial for everyone. One person who was at the forefront of this movement was a young woman named Maya. Maya was a brilliant and ambitious researcher who had dedicated her life to understanding the implications of emerging technologies like artificial intelligence and biotechnology. She was deeply concerned about the potential risks and unintended consequences of these technologies, and she worked tirelessly to raise awareness about the need for responsible innovation. Maya's work had earned her a reputation as one of the most influential voices in the field of technology and ethics, and she was widely respected for her deep understanding of the issues and her ability to communicate complex ideas in ways that were accessible and engaging. She was also known for her passionate and inspiring speeches, which often left her audiences with a sense of purpose and determination to make the world a better place through their own efforts. One day, Maya received an invitation to speak at a major conference on technology and ethics, which was being held in a large convention center in the heart of the city. The conference was expected to attract thousands of people from all over the world, and there was a great deal of excitement and anticipation about what Maya would say. As she prepared for her speech, Maya knew that she had a big responsibility on her shoulders. She felt a deep sense of obligation to use her platform to inspire others to take action and make a difference in the world, and she was determined to do everything in her power to live up to this responsibility. When the day of the conference arrived, Maya was filled with a mixture of excitement and nerves. She spent hours rehearsing her speech and fine-tuning her ideas, making sure that she had everything just right. Finally, after what felt like an eternity, it was time for her to take the stage. As she stepped up to the podium, Maya could feel the energy of the crowd surging around her. She took a deep breath and began to speak, her voice strong and clear as she outlined the challenges and opportunities facing society in the age of technology. She spoke passionately about the need for responsible innovation and the importance of considering the ethical implications of our actions, and she inspired many people in the audience to take up this cause and make a difference in their own lives. Overall, Maya's speech was a resounding success, and she received countless messages of gratitude and appreciation from those who had heard her speak. She knew that there was still much work to be done, but she felt hopeful about the future and the role that technology could play in creating a better world for all.  As Maya left the stage and made her way back to her seat, she couldn't help but feel a sense of pride and accomplishment at what she had just accomplished. She knew that her words had the power to inspire others and make a real difference in the world, and she was grateful for the opportunity to have played a part in this important work. For Maya, the future was full of promise and possibility, and she was determined to continue doing everything in her power to help create a brighter, more ethical world for everyone. As long as you love her, nothing gona change his love for her. You are her sunshine, the "]


# PROMPTS = ["模仿李白的风格写一首七律·飞机"]
# PROMPTS = ["帮我用HTML设计一个好玩的贪吃蛇游戏，800x600大小，适合新手玩的"]
#PROMPTS = ["At a restaurant, each adult meal costs $5 and kids eat free. If a group of 15 people came in and 8 were kids, how much would it cost for the group to eat?"]


def print_health() -> None:
    """Prints server health."""
    with requests.get(
        f"{LLM_SERVER_URL}/health", timeout=(CONNECT_TIMEOUT, RESPONSE_TIMEOUT)
    ) as response:
        print(response.text)


def task(id_: int) -> None:
    """Basic query task.

    Args:
        id_ (int): task id
    """
    payload: dict = {
        "model": "",
        "messages": [
            {
                "role": "user",
                "content": f"{random.choice(PROMPTS)}\n",
            },
        ],
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }
    try:
        with requests.post(
            f"{LLM_SERVER_URL}/v1/chat/completions",
            json=payload,
            timeout=(CONNECT_TIMEOUT, RESPONSE_TIMEOUT),
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue

                if line.startswith(b"data: {"):
                    delta: dict = json.loads(line[len("data: ") :])["choices"][0]["delta"]
                    if "content" in delta and delta["content"]:
                        with Path(f"task-{id_}.txt").open("a", encoding="utf-8") as file:
                            file.write(delta["content"])

    except requests.Timeout:
        with Path(f"task-{id_}.txt").open("w", encoding="utf-8") as file:
            file.write(f"Timeout after {RESPONSE_TIMEOUT} seconds")


if __name__ == "__main__":
    processes: list[tuple[int, Process]] = [
        (i, Process(target=task, args=(i,))) for i in range(NUM_INSTANCES)
    ]
    for i, process in processes:
        process.start()
        print("Task", i, "started")

    time.sleep(1.0)
    print("0 : ", end="")
    print_health()

    elapsed_seconds: int = 0
    completed: list[bool] = [False] * len(processes)
    while True:
        if all(completed):
            break

        for i, process in processes:
            if not process.is_alive() and not completed[i]:
                print("Task ", i, " finished\n", elapsed_seconds, end=" : ", sep="")
                print_health()

        completed = [not thread.is_alive() for _i, thread in processes]

        time.sleep(1)
        elapsed_seconds += 1

        if elapsed_seconds % 30 == 0:
            print(elapsed_seconds, end=" : ")
            print_health()

        if elapsed_seconds > MAXIMUM_RUNTIME:
            print(f"Time limit of {MAXIMUM_RUNTIME} seconds reached, killing running tasks")
            for i, process in processes:
                if process.is_alive():
                    print("Killing task", i)
                    process.terminate()
            break

    with Path("tasks.txt").open("w", encoding="utf-8") as file:
        for id_, _process in processes:
            task_file_path = Path(f"task-{id_}.txt")

            if not task_file_path.exists():
                file.write(f"# TASK {id_}\n\n")
                file.write(f"Task failed to receive response within {MAXIMUM_RUNTIME} seconds")
                file.write("\n\n")
                continue

            with task_file_path.open("r", encoding="utf-8") as task_file:
                file.write(f"# TASK {id_}\n\n")
                file.write(task_file.read())
                file.write("\n\n")

            task_file_path.unlink()
