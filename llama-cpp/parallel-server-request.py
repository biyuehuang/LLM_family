"""
Parallel server request test script.
    usage: python3 parallel-server-request.py [num_instances]
"""

import json
import time
import sys
import random
from multiprocessing import Process
from pathlib import Path
from typing import Final

import requests

LLM_SERVER_URL: Final[str] = "http://localhost:8080"
NUM_INSTANCES: Final[int] = int(sys.argv[1]) if len(sys.argv) > 1 else 1
CONNECT_TIMEOUT: float = 300.0
RESPONSE_TIMEOUT: float = 300.0
MAXIMUM_RUNTIME: int = 3000
MAX_TOKENS: int = 100

PROMPTS = [
    "What is the meaning of life?\n",
    "Explain the theory of relativity.\n",
    "Describe the process of photosynthesis.\n",
    "What are the main causes of climate change?\n",
    "How does quantum computing work?\n",
    "What is the significance of the Turing test?\n",
    "Discuss the impact of artificial intelligence on society.\n",
    "What are the ethical implications of genetic engineering?\n",
]

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
        "max_tokens": 100,
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
