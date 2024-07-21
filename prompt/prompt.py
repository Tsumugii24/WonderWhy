import os
import sys
import json

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录的绝对路径
PROMPT_FILE_PATH = os.path.join(ROOT_PATH, 'prompt.json')  # Prompt文件的绝对路径


class Prompt:
    def __init__(self, prompt_file):
        with open(prompt_file, 'r', encoding="utf-8") as f:
            prompt = json.load(f)
        self.system_prompt = prompt['system_prompt']
        self.adult_system_prompt = prompt['adult_system_prompt']
        self.child_system_prompt = prompt['child_system_prompt']
        self.student_system_prompt = prompt['student_system_prompt']
        self.user_prompt = prompt['user_prompt']
        self.poet = prompt['poet']
        self.image_prompt = prompt['image_prompt']  # 产生生图Prompt的Prompt


# 创建一个Prompt实例
prompt = Prompt(PROMPT_FILE_PATH)

if __name__ == '__main__':
    print(prompt.system_prompt)
    print(prompt.default_system_prompt)
    print(prompt.child_system_prompt)
    print(prompt.student_system_prompt)
    print(prompt.user_prompt)
    print(prompt.poet)
