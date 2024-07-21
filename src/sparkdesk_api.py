# coding: utf-8
from loguru import logger   # 用于日志记录
import time
import os

import requests
from datetime import datetime
from wsgiref.handlers import format_date_time
from time import mktime
import hashlib
import base64
import hmac
from urllib.parse import urlencode
import json
from PIL import Image
from io import BytesIO

import _thread as thread
from urllib.parse import urlparse
import ssl
import websocket
import openpyxl


""" 生图 """

class AssembleHeaderException(Exception):
    def __init__(self, msg):
        self.message = msg

class Url:
    def __init__(this, host, path, schema):
        this.host = host
        this.path = path
        this.schema = schema
        pass

# calculate sha256 and encode to base64
def sha256base64(data):
    sha256 = hashlib.sha256()
    sha256.update(data)
    digest = base64.b64encode(sha256.digest()).decode(encoding='utf-8')
    return digest

def parse_url(requset_url):
    stidx = requset_url.index("://")
    host = requset_url[stidx + 3:]
    schema = requset_url[:stidx + 3]
    edidx = host.index("/")
    if edidx <= 0:
        raise AssembleHeaderException("invalid request url:" + requset_url)
    path = host[edidx:]
    host = host[:edidx]
    u = Url(host, path, schema)
    return u

# 生成鉴权url
def assemble_ws_auth_url(requset_url, method="GET", api_key="", api_secret=""):
    u = parse_url(requset_url)
    host = u.host
    path = u.path
    now = datetime.now()
    date = format_date_time(mktime(now.timetuple()))
    # print(date)
    # date = "Thu, 12 Dec 2019 01:57:27 GMT"
    signature_origin = "host: {}\ndate: {}\n{} {} HTTP/1.1".format(host, date, method, path)
    # print(signature_origin)
    signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                             digestmod=hashlib.sha256).digest()
    signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
    authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
        api_key, "hmac-sha256", "host date request-line", signature_sha)
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
    # print(authorization_origin)
    values = {
        "host": host,
        "date": date,
        "authorization": authorization
    }

    return requset_url + "?" + urlencode(values)

# 生成请求body体
def getBody(appid,text):
    body= {
        "header": {
            "app_id": appid,
            "uid":"123456789"
        },
        "parameter": {
            "chat": {
                "domain": "general",
                "temperature":0.5,
                "max_tokens":4096
            }
        },
        "payload": {
            "message":{
                "text":[
                    {
                        "role":"user",
                        "content":text
                    }
                ]
            }
        }
    }
    return body


""" GPT """


# class WS(websocket.WebSocketApp):
#     """
#     WebSocketApp子类，添加消息变量
#     """

#     def __init__(self, appid, url, on_message, on_error, on_close, on_open):
#         self.appid = appid
#         self.received_message = ""
#         super(WS, self).__init__(url=url,
#                                  on_message=on_message,
#                                  on_error=on_error,
#                                  on_close=on_close,
#                                  on_open=on_open)

# class Ws_Param(object):
#     # 初始化
#     def __init__(self, APPID, APIKey, APISecret, gpt_url):
#         self.APPID = APPID
#         self.APIKey = APIKey
#         self.APISecret = APISecret
#         self.host = urlparse(gpt_url).netloc
#         self.path = urlparse(gpt_url).path
#         self.gpt_url = gpt_url

#     def create_signature_origin(self, date) -> str:
#         """ 拼接字符串,得到初始签名signature_origin """
#         signature_origin = "host: " + self.host + "\n"
#         signature_origin += "date: " + date + "\n"
#         signature_origin += "GET " + self.path + " HTTP/1.1"    # GPT交互使用Get
#         return signature_origin
    
#     # 生成url
#     def create_url(self):
#         """
#         生成url [host主机 + date时间戳(RFC1123格式) + authorization认证信息]
#         """
#         # 生成RFC1123格式的时间戳
#         now = datetime.now()
#         date = format_date_time(mktime(now.timetuple()))

#         signature_origin = self.create_signature_origin(date)
#         # 进行hmac-sha256进行加密
#         signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
#                                  digestmod=hashlib.sha256).digest()

#         signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

#         authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

#         authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

#         # 将请求的鉴权参数组合为字典
#         v = {
#             "authorization": authorization,
#             "date": date,
#             "host": self.host
#         }
#         # 拼接鉴权参数，生成url
#         url = self.gpt_url + '?' + urlencode(v)
#         # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
#         return url


# # 收到websocket错误的处理
# def on_error(ws, error):
#     print("### error:", error)


# # 收到websocket关闭的处理
# def on_close(ws):
#     print("### closed ###")


# # 收到websocket连接建立的处理
# def on_open(ws):
#     thread.start_new_thread(run, (ws,))


# def run(ws, *args):
#     data = json.dumps(gen_params(appid=ws.appid, query=ws.query, domain=ws.domain))
#     ws.send(data)


# # 收到websocket消息的处理
# def on_message(ws, message):
#     # print(message)
#     data = json.loads(message)
#     code = data['header']['code']
#     if code != 0:
#         print(f'请求错误: {code}, {data}')
#         ws.close()
#     else:
#         choices = data["payload"]["choices"]
#         status = choices["status"]
#         content = choices["text"][0]["content"]
#         print(content,end='')
#         if status == 2:
#             print("#### 关闭会话")
#             ws.close()


# def gen_params(appid, query, domain):
#     """
#     通过appid和用户的提问来生成请参数
#     """

#     data = {
#         "header": {
#             "app_id": appid,
#             "uid": "1234",           
#             # "patch_id": []    #接入微调模型，对应服务发布后的resourceid          
#         },
#         "parameter": {
#             "chat": {
#                 "domain": domain,
#                 "temperature": 0.5,
#                 "max_tokens": 4096,
#                 "auditing": "default",
#             }
#         },
#         "payload": {
#             "message": {
#                 "text": [
#                     {
#                         "role": "user", 
#                         "content": query
#                     }
#                 ]
#             }
#         }
#     }
#     return data


# def call(appid, api_secret, api_key, gpt_url, domain, query) -> str:
#     wsParam = Ws_Param(appid, api_key, api_secret, gpt_url)
#     websocket.enableTrace(False)
#     wsUrl = wsParam.create_url()

#     ws = WS(appid=wsParam.APPID,
#             url=wsUrl, 
#             on_message=on_message, 
#             on_error=on_error, 
#             on_close=on_close, 
#             on_open=on_open)
#     ws.appid = appid
#     ws.query = query
#     ws.domain = domain
#     ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
#     return ws.received_message


# if __name__ == "__main__":
#     call( 
#         appid="",
#         api_secret="",
#         api_key="",
#         #appid、api_secret、api_key三个服务认证信息请前往开放平台控制台查看（https://console.xfyun.cn/services/bm35）
#         gpt_url="wss://spark-api.xf-yun.com/v3.5/chat",      # Max环境的地址   
#         # Spark_url = "ws://spark-api.xf-yun.com/v3.1/chat"  # Pro环境的地址
#         # Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat"  # Lite环境的地址
#         domain="generalv3.5",     # Max版本
#         # domain = "generalv3"    # Pro版本
#         # domain = "general"      # Lite版本址
#         query="给我写一篇100字的作文"
#     )
