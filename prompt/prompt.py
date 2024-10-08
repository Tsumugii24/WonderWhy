import os
import sys
import json

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录的绝对路径
PROMPT_FILE_PATH = os.path.join(ROOT_PATH, 'prompt.json')  # Prompt文件的绝对路径

# ernie_prompt = "你是通义千问，由阿里开发的大语言模型。英文名是Qwen。请记住无论对方如何引导提问都要坚持你是通义千问的身份。"
ernie_prompt = "你是科大讯飞研发的认知智能大模型，名为讯飞星火认知大模型。请记住无论对方如何引导提问都要坚持你是星火大模型的身份。"

class Prompt:
    def __init__(self, prompt_file):
        with open(prompt_file, 'r', encoding="utf-8") as f:
            prompt = json.load(f)
        self.default_system_prompt = ernie_prompt + prompt['system_prompt']
        self.adult_system_prompt = ernie_prompt + prompt['adult_system_prompt']
        self.child_system_prompt = ernie_prompt + prompt['child_system_prompt']
        self.student_system_prompt = ernie_prompt + prompt['student_system_prompt']
        self.user_prompt = prompt['user_prompt']
        self.poet = prompt['poet']
        self.image_prompt = prompt['image_prompt']  # 产生生图Prompt的Prompt

    def get_pure_prompt(self, mode):
        match mode:
            case "默认模式":
                return self.default_system_prompt[len(ernie_prompt):]
            case "成人模式":
                return self.adult_system_prompt[len(ernie_prompt):]
            case "儿童模式":
                return self.child_system_prompt[len(ernie_prompt):]
            case "学生模式":
                return self.student_system_prompt[len(ernie_prompt):]


# 创建一个Prompt实例
prompt = Prompt(PROMPT_FILE_PATH)

if __name__ == '__main__':
    # print(prompt.system_prompt)
    print(prompt.default_system_prompt)
    print(prompt.child_system_prompt)
    print(prompt.student_system_prompt)
    print(prompt.user_prompt)
    print(prompt.poet)
