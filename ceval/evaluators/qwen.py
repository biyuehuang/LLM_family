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
# refer to https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_chat_ceval.py

import re
from tqdm import tqdm
#import torch
from thefuzz import process
#from transformers import AutoTokenizer
#from transformers.generation import GenerationConfig

#from ipex_llm.transformers import AutoModelForCausalLM, convert_model_hybrid
from evaluators.evaluator import Evaluator

import requests

# 先打开flowy,再运行
base_url = "http://localhost:11434/api/generate"


class QwenEvaluator(Evaluator):
    def __init__(self, choices, model_path, device="GPU",qtype="int4"):
        super(QwenEvaluator, self).__init__(choices, model_path, device,qtype)   
        self.model_path = model_path

    def get_model_response(self,prompt):
    # """
    # 向 ollama API 发送请求并获取模型的响应
    # :param prompt: 用户输入的问题
        # :return: 模型的响应文本
        # """
        data = {
            "model": self.model_path,
            "prompt": prompt
        }
        try:
            # 发送 POST 请求到 ollama API
            response = requests.post(base_url, json=data)
            # 检查响应状态码
            response.raise_for_status()
            # 逐行拼接响应内容
            full_response = ""
            for line in response.text.strip().split('\n'):
                if line:
                    import json
                    line_data = json.loads(line)
                    full_response += line_data.get("response", "")
            return full_response
        except requests.exceptions.RequestException as e:
            print(f"请求出错: {e}")
            return None


        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.model_path,
        #     trust_remote_code=True
        # )
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_path,
        #     load_in_4bit=True,
        #     #load_in_low_bit=self.qtype,
        #     optimize_model=True,
        #     use_cache=True,
        #     trust_remote_code=True
        # )
        # #self.model = self.model.bfloat16()
        # #self.model = convert_model_hybrid(self.model)
        # self.model = self.model.eval().to(self.device)
        # # self.model = AutoModelForCausalLM.load_low_bit(
        # #     self.model_path,
        # #     use_cache=True,
        # #     trust_remote_code=True
        # # ).eval().to(self.device)

        # self.model.generation_config = GenerationConfig.from_pretrained(
        #     self.model_path,
        #     trust_remote_code=True
        # )
        # self.model.generation_config.do_sample = False  # use greedy decoding
        # self.model.generation_config.repetition_penalty = 1.0  # disable repetition penalty

    def compare_ab(self,last_answer_tag_index,last_answer_text_index):
        # 若都未找到
        if last_answer_tag_index == -1 and last_answer_text_index == -1:
            print("未找到 <answer> 和 答案")
            return -1
        # 若仅 <answer> 未找到
        elif last_answer_tag_index == -1:
            print(f"位置最靠前的是 答案，位置是 {last_answer_text_index}")
            return last_answer_text_index
        # 若仅 答案 未找到
        elif last_answer_text_index == -1:
            print(f"位置最靠前的是 <answer>，位置是 {last_answer_tag_index}")
            return last_answer_tag_index
        # 若都找到，比较位置
        else:
            if last_answer_tag_index < last_answer_text_index:
                print(f"位置最靠前的是 <answer>，位置是 {last_answer_tag_index}")
                return last_answer_tag_index
            else:
                print(f"位置最靠前的是 答案，位置是 {last_answer_text_index}")
                return last_answer_text_index

    def process_before_extraction2(self, text):
        start_index = text.rfind('</think>')
        if start_index != -1:

            text = text[start_index :]
        else:
            text = text
            
        # 查找最后一个 <answer> 的位置
        last_answer_tag_index = text.rfind('<answer>')

        # 查找最后一个 答案 的位置
        last_answer_text_index = text.rfind('答案')

        start_index = self.compare_ab(last_answer_tag_index,last_answer_text_index)

        if start_index != -1: # and end_index != -1:
            answer = text[start_index :]
           # print("answer",answer)
            return answer
        else:
            return text

    def process_before_extraction(self, gen, question, choice_dict):

        question_split = question.rstrip("。").split("。")[-1].split("_")

        if len(question_split[0].strip()) > 4:
            gen = gen.replace(question_split[0], "答案是")
        if len(question_split[-1].strip()) > 4:
            gen = gen.replace(question_split[-1], "")

        for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
            gen = gen.replace(val.rstrip("。"), key)
        return gen


    def count_substr(self, gen, pattern):
        return len(re.findall(pattern, gen))


    def extract_choice(self, gen, prompt, choice_list):
        res = re.search(
            r"(?:(?:选|选择|选定)[：:]?\s*|(?:(?:答案|选项)(?![^ABCD]{0,10}?(?:不|非)[^ABCD]{0,10}?(?:是|选|为|：|:|】))[^ABCD]{0,10}?(?:是|选|为|：|:|】))[^ABCD]{0,10}?)(A|B|C|D)(?:选项)?(?:\)|。|\.|，|,|．|、|A|B|C|D|$|：|:|\)|）)",
            gen,
        )

        if res is None:
            res = re.search(
                r"(A|B|C|D)(?:选?项)?(?![^ABCD]{0,4}?(?:不|非)[^ABCD]{0,4}?(?:正确|对[的，。：]|符合))[^ABCD]{0,4}?(?:正确|对[的，。：]|符合)",
                gen,
            )

        if res is None:
            res = re.search(r"^[\(（]?(A|B|C|D)(?:。|\)|）|\.|，|,|．|：|:|$)", gen)

        if res is None:
            res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

        if res is None:
            return self.choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
        return res.group(1)


    def format_example(self, line):
        example = line["question"] + "\n\n"
        for choice in self.choices:
            example += f'{choice}. {line[f"{choice}"]}\n'
        return example


    def extract_answer(self, response, row):
        prompt = row["question"]
        gen = self.process_before_extraction2(
            response
        )
        if not isinstance(prompt, str):
            prompt = prompt[0]

        #print("*********gen",gen)
        #print("*********prompt",prompt)
        pred = self.extract_choice(gen, prompt, [row[choice] for choice in self.choices])
        return pred


    #@torch.no_grad()
    def eval_subject(
        self,
        subject_name,
        test_df,
        eval_type="validation" # "test","validation"
    ):
        if eval_type == "validation":
            responses = []
            result = []
            score = []
            # A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
            PROMPT_FORMAT = """
            以下是中国关于科目{subject_name}考试的单项选择题，请选出其中的正确答案。The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> 答案是 </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> 答案是 answer here </answer>.
            User: {prompt}.
            Assistant: <think>
            """

            for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
                question = self.format_example(row)
  
                in_prompt = PROMPT_FORMAT.format(subject_name=subject_name,prompt=question)

                response = self.get_model_response(in_prompt)

                pred = self.extract_answer(response, row)
                #print("********question:\n",question)
                
                print("********response:\n",response)
                print("********pred:\n",pred)
                print("********row:\n",row)
                if "answer" in row:
                    correct = 1 if pred == row["answer"] else 0
                    score.append(correct)
                responses.append(response)
                result.append(pred)

            if score:
                correct_ratio = 100 * sum(score) / len(score)

            else:
                correct_ratio = 0

            return correct_ratio, None
        elif eval_type == "test":
            answers = {}
            for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
                question = self.format_example(row)
                response, _ = self.model.chat(
                    self.tokenizer,
                    question,
                    history=None,
                )
                pred = self.extract_answer(response, row)
                answers[str(i)] = pred
            return None, answers
