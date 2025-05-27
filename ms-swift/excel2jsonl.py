import pandas as pd
import json
import copy

# 假设这是你的常量
SYSTEM = "你是人工智能助手，三观正直。可以通过xx来回复用户的问题，还能进行知识问答。"

# 假设这是你的模板
template = {"id": "Light_Train_<id>",
            "conversations": [{"from": "system", "value": "SYSTEM"},
                              {"from": "user", "value": "<prompt>"}, 
                              {"from": "assistant", "value": "<response>"}
            ]
            }

# 读取 train.xlsx 文件
try:
    df = pd.read_excel('train.xlsx')
except FileNotFoundError:
    print("未找到 train.xlsx 文件，请检查文件路径。")
    exit(1)

# 打开一个文件用于写入 jsonl 数据
with open('train.jsonl', 'w', encoding='utf-8') as f:
    # 遍历 DataFrame 的每一行
    for index, row in df.iterrows():
        print(index, row)
        print("--------------------------")
        # 复制模板，避免修改原始模板
        filled_template = copy.deepcopy(template)
        # 填充 <id>
        filled_template["id"] = filled_template["id"].replace("<id>", str(index))
        # 填充 API_LIST
       # if "tools" in filled_template:
       #     filled_template["tools"] = filled_template["tools"].replace("API_LIST", json.dumps(API_LIST))
        # 填充 SYSTEM
        for conversation in filled_template["conversations"]:
            #print(conversation)
           # print(conversation["from"])
            if "from" in conversation and conversation["from"] == "system":
                conversation["value"] = conversation["value"].replace("SYSTEM", SYSTEM)
      
        # 填充 <prompt>
        for conversation in filled_template["conversations"]:
            #print("*********",row['Request'])
            if "from" in conversation and conversation["from"] == "user":
                #print("*********",row['Request'])
                conversation["value"] = conversation["value"].replace("<prompt>", str(row['Request']))
        # 填充 <response>
        for conversation in filled_template["conversations"]:
            if "from" in conversation and conversation["from"] == "assistant":
                conversation["value"] = conversation["value"].replace("<response>", str(row['Response']))

        # 将填充后的模板转换为 JSON 字符串并写入文件
        json_str = json.dumps(filled_template, ensure_ascii=False)
        f.write(json_str + '\n')

print("数据已成功保存到 train.jsonl 文件。")


template_val = {"id": "Light_Val_<id>",
            "conversations": [{"from": "system", "value": "SYSTEM"},
                              {"from": "user", "value": "<prompt>"}, 
                              {"from": "assistant", "value": "<response>"}
            ]
            }

# 读取 val.xlsx 文件
try:
    df = pd.read_excel('val.xlsx')
except FileNotFoundError:
    print("未找到 val.xlsx 文件，请检查文件路径。")
    exit(1)

# 打开一个文件用于写入 jsonl 数据
with open('val.jsonl', 'w', encoding='utf-8') as f:
    # 遍历 DataFrame 的每一行
    for index, row in df.iterrows():
        print(index, row)
        print("--------------------------")
        # 复制模板，避免修改原始模板
        filled_template = copy.deepcopy(template_val)
        # 填充 <id>
        filled_template["id"] = filled_template["id"].replace("<id>", str(index))
        # 填充 API_LIST
       # if "tools" in filled_template:
       #     filled_template["tools"] = filled_template["tools"].replace("API_LIST", json.dumps(API_LIST))
        # 填充 SYSTEM
        for conversation in filled_template["conversations"]:
            #print(conversation)
           # print(conversation["from"])
            if "from" in conversation and conversation["from"] == "system":
                conversation["value"] = conversation["value"].replace("SYSTEM", SYSTEM)
      
        # 填充 <prompt>
        for conversation in filled_template["conversations"]:
            #print("*********",row['Request'])
            if "from" in conversation and conversation["from"] == "user":
                print("*********",row['Request'])
                conversation["value"] = conversation["value"].replace("<prompt>", str(row['Request']))
        # 填充 <response>
        for conversation in filled_template["conversations"]:
            if "from" in conversation and conversation["from"] == "assistant":
                conversation["value"] = conversation["value"].replace("<response>", str(row['Response']))

        # 将填充后的模板转换为 JSON 字符串并写入文件
        json_str = json.dumps(filled_template, ensure_ascii=False)
        f.write(json_str + '\n')

print("数据已成功保存到 val.jsonl 文件。")
