import os
import json


if __name__ == "__main__":
    # import openai

    # # openai.api_base = "https://api.openai.com/v1" # 换成代理，一定要加 v1
    # openai.api_base = "https://apikeyplus.com/v1" # 换成代理，一定要加 v1
    # # openai.api_key = "API_KEY"
    # openai.api_key = "sk-wCtvywxzqFSvoYX30e499594Ee814425881b96C0459dF4Be"

    # for resp in openai.ChatCompletion.create(
    #                                     model="gpt-3.5-turbo",
    #                                     messages=[
    #                                     {"role": "user", "content": "说你好"}
    #                                     ],
    #                                     # 流式输出
    #                                     stream = True):
    #     if 'content' in resp.choices[0].delta:
    #         print(resp.choices[0].delta.content, end="", flush=True)
    
    # 导入SDK，发起请求
    # from openai import OpenAI
    # client = OpenAI(
    #                 # 控制台获取key和secret拼接，假使APIKey是key123456，APISecret是secret123456
    #         api_key="e0799f95d4cd84750874060b892943e0:M2Q3ZjllY2JlODE2MzlmNGYyYjVkZGRm", 
    #         base_url = 'https://spark-api-open.xf-yun.com/v1/' # 指向讯飞星火的请求地址
    #     )
    # completion = client.chat.completions.create(
    #     model='generalv3.5', # 指定请求的版本
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": '你是谁'
    #         }
    #     ]
    # )
    # print(completion.choices[0].message)


    # from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
    # from sparkai.core.messages import ChatMessage

    # #星火认知大模型Spark Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
    # SPARKAI_URL = 'ws(s)://spark-api.xf-yun.com/v2.1/chat'
    # #星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
    # SPARKAI_APP_ID = 'c96bdfdd'
    # SPARKAI_API_SECRET = 'M2Q3ZjllY2JlODE2MzlmNGYyYjVkZGRm'
    # SPARKAI_API_KEY = 'e0799f95d4cd84750874060b892943e0'
    # #星火认知大模型Spark Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
    # SPARKAI_DOMAIN = 'generalv2'

    # if __name__ == '__main__':
    #     spark = ChatSparkLLM(
    #         spark_api_url=SPARKAI_URL,
    #         spark_app_id=SPARKAI_APP_ID,
    #         spark_api_key=SPARKAI_API_KEY,
    #         spark_api_secret=SPARKAI_API_SECRET,
    #         spark_llm_domain=SPARKAI_DOMAIN,
    #         streaming=False,    # callbacks不起作用
    #     )
    #     messages = [ChatMessage(
    #         role="user",
    #         content='你好呀'
    #     )]
    #     handler = ChunkPrintHandler()
    #     a = spark.generate([messages], callbacks=[handler])
    #     print(a)
    #     print(a.generations[0][0].text)
    #     print(a.llm_output)
    
    # from src.gen_image import image_generator
    # desc = '''生成一张图：远处有着高山，山上覆盖着冰雪，近处有着一片湛蓝的湖泊'''
    # image_generator.generate_image(desc)

    from src.config import bing_search_api_key
    from src.search_engine import search_with_bing
    from src.search_engine import search_with_duckduckgo
    print("====DUCKDUCKGO=====")
    print(search_with_duckduckgo("静夜思"))
    print("====BING=====")
    print(search_with_bing("静夜思", bing_search_api_key))
