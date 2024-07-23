import json
import os

def set_api_key_from_json(json_file_path):
    try:
        # 打开并读取JSON文件
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        # 从JSON数据中获取api_key
        api_key= data.get('openai_api_key','sk-')
        api_base= data.get('openai_api_base','https://apikeyplus.com/v1')
        model_name= data.get('default_model','gpt-3.5-turbo')

        # 设置环境变量
        os.environ['openai_api_key'] = api_key
        os.environ['openai_api_base'] = api_base
        os.environ['default_model'] = model_name
    except FileNotFoundError:
        print(f"The file {json_file_path} does not exist.")
    except json.JSONDecodeError:
        print("Error decoding the JSON file.")
    except Exception as e:
        print(f"An error occurred: {e}")