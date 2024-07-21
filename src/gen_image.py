# -*- coding: utf-8 -*-
from loguru import logger   # 用于日志记录
import time
import os
import requests
import base64
import json
from PIL import Image
from io import BytesIO
import ssl
import websocket

from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

from src.sparkdesk_api import (
    assemble_ws_auth_url,
    getBody,
)


from src.config import (
    sparkdesk_appid,
    sparkdesk_apikey,
    sparkdesk_apisecret
)
APPID = sparkdesk_appid
APIKEY = sparkdesk_apikey
APISECRET = sparkdesk_apisecret

from prompt.prompt import prompt
IMAGE_PROMPT = prompt.image_prompt  # 产生生图Prompt的Prompt

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
IMAGE_DIR = os.path.join(PROJECT_DIR, "images")


class PromptGenerator:
    ''' 生成生图的prompt '''
    def __init__(self, appid, apikey, apisecret):
        self.appid = appid
        self.apikey = apikey
        self.apisecret = apisecret
        self.host = "ws(s)://spark-api.xf-yun.com/v2.1/chat" # V2.0
        self.domain = "generalv2"       # V2.0
        self.spark = ChatSparkLLM(
            spark_api_url=self.host,
            spark_app_id=appid,
            spark_api_key=apikey,
            spark_api_secret=apisecret,
            spark_llm_domain=self.domain,
            streaming=False,    # callbacks不起作用
        )
        
    
    def generate_prompt(self, poet):
        ''' 生成用于生图的prompt '''
        query = IMAGE_PROMPT.format(poet)    # 加载生图prompt，并与诗句拼接
        # 请求SparkDesk
        messages = [ChatMessage(
            role='system', 
            content=query
        )]
        handler = ChunkPrintHandler()
        response = self.spark.generate([messages], callbacks=[handler])
        # 提取数据
        tokens = response.llm_output['token_usage']['total_tokens']
        logger.debug(f"生成生图Prompt,SparkDesk消耗tokens: {tokens}")
        prompt = response.generations[0][0].text
        return prompt
    

class ImageGenerator:
    ''' 文生图 '''
    def __init__(self, appid, apikey, apisecret):
        print(appid)
        print(apikey)
        print(apisecret)
        self.appid = appid
        self.apikey = apikey
        self.apisecret = apisecret
        self.host = 'http://spark-api.cn-huabei-1.xf-yun.com/v2.1/tti'
        self.prompt_generator = PromptGenerator(
            appid=appid, 
            apikey=apikey, 
            apisecret=apisecret
        )

    def base64_to_image(self, base64_data, save_path):
        '''将base64的图片数据存在本地'''
        # 解码base64数据
        img_data = base64.b64decode(base64_data)
        # 将解码后的数据转换为图片
        img = Image.open(BytesIO(img_data))
        # 保存图片到本地
        img.save(save_path)

    def generate_image(self, prompt):
        if(prompt=="本地"):
            return self.generate_local_image(prompt)
        img_path = self.generate_image_from_text(poet=prompt)
        return img_path

    def generate_image_from_text(self, poet): # width=1024, height=1024, image_num=1):
        ''' 基于讯飞星火API生图 '''
        # 生成生图prompt
        # prompt = self.prompt_generator.generate_prompt(poet)
        prompt = poet

        logger.debug("发送生图https请求")
        # 发起https请求
        url = assemble_ws_auth_url(self.host, 
                                   method='POST', 
                                   api_key=self.apikey, 
                                   api_secret=self.apisecret)
        content = getBody(self.appid, prompt)
        begin_time = time.time()
        response = requests.post(url,
                                 json=content,
                                 headers={'content-type': "application/json"}).text
        end_time = time.time()
        logger.debug(f"生成生图耗时: {end_time - begin_time}, 解析https数据")
        # 解析数据
        data = json.loads(response)
        # print("data" + str(response))
        code = data['header']['code']
        if code != 0:
            logger.error(f"请求错误: {code}, {data}")
        else:
            text = data["payload"]["choices"]["text"]
            imageContent = text[0]
            # if('image' == imageContent["content_type"]):
            imageBase = imageContent["content"]
            imageName = data['header']['sid']

            logger.debug("解码图片")
            # 解码图片并保存至本地
            savePath = os.path.join(IMAGE_DIR, f"{imageName}.jpg") # f"D://{imageName}.jpg"
            self.base64_to_image(imageBase, savePath)   # BUG 显示传入3个参数，图像成功生成
            logger.debug(f"图片保存路径: {savePath}")
        return savePath
        # '''借助Appbuilder生图'''
        # import appbuilder
        # os.environ["APPBUILDER_TOKEN"] = "bce-v3/ALTAK-QwuihwYsMjA5jBiIBVfJP/51f962e086efb6f3a2332414360552bae5f3958d"
        # text2Image = appbuilder.Text2Image()
        # content_data = {"prompt": prompt, "width": width, "height": height, "image_num": image_num}
        # msg = appbuilder.Message(content_data)
        # out = text2Image.run(msg)
        # return out.content['img_urls']

    def generate_local_image(self):
        ''' 直接给出本地示例图片`generated_image.png` '''
        local_image_path = os.path.join("images", "generate_image.png")
        
        # 模拟生成图片保存
        # 这里你可以替换为实际的图像生成逻辑
        # from PIL import Image, ImageDraw, ImageFont
        # image = Image.new('RGB', (1024, 1024), color = (73, 109, 137))
        # d = ImageDraw.Draw(image)
        # d.text((10,10), prompt, fill=(255,255,0))
        # image.save(local_image_path)
        time.sleep(2)
        return local_image_path


image_generator = ImageGenerator(appid=APPID,
                                 apisecret=APISECRET,
                                 apikey=APIKEY)

if __name__ == '__main__':
    desc = '''生成一张图：远处有着高山，山上覆盖着冰雪，近处有着一片湛蓝的湖泊'''
    image_generator.generate_image(desc)
    # print(IMAGE_PROMPT)