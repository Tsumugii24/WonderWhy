
import base64
import datetime
import json
import os
from io import BytesIO
import requests

import gradio as gr
import requests
from PIL import Image
from loguru import logger
from zhipuai import ZhipuAI
from loguru import logger

from src import shared, config
from src.base_model import BaseLLMModel
from src.presets import (
    INITIAL_SYSTEM_PROMPT,
    TIMEOUT_ALL,
    TIMEOUT_STREAMING,
    STANDARD_ERROR_MSG,
    CONNECTION_TIMEOUT_MSG,
    READ_TIMEOUT_MSG,
    ERROR_RETRIEVE_MSG,
    GENERAL_ERROR_MSG,
    CHAT_COMPLETION_URL,
    SUMMARY_CHAT_SYSTEM_PROMPT
)
from src.openai_client import OpenAIClient

from src.utils import (
    count_token,
    construct_system,
    construct_user,
    get_last_day_of_month,
    i18n,
    replace_special_symbols,
)




def decode_chat_response(response):
    try:
        error_msg = ""
        for chunk in response:
            if chunk:
                # chunk = chunk.decode()
                chunk = chunk.choices[0].delta
                chunk_length = len(chunk.content)
                try:
                    if chunk_length > 1 and chunk!="":
                        try:
                            yield chunk.content
                        except Exception as e:
                            logger.error(f"Error xxx: {e}")
                            continue
                except Exception as ee:
                    logger.error(f"ERROR: {chunk}, {ee}")
                    continue
        if error_msg and not error_msg.endswith("[DONE]"):
            raise Exception(error_msg)
    except GeneratorExit as ge:
        raise ValueError(f"GeneratorExit: {ge}")
    except Exception as e:
        raise Exception(f"Error in generate: {str(e)}")



def query_knowledge_base(query, kb_name, only_imgs = False):
    """
    Call the FastAPI endpoint to query the knowledge base.

    Args:
        query (str): The query to send to the knowledge base.
        kb_name (str): The name of the knowledge base.
    
    Returns:
        dict: The response from the API.
    """
    url = "http://127.0.0.1:8000/v1/kb/chat_kb"  # Adjust the URL to your FastAPI server

    data = {
        "query": query,
        "kb_name": kb_name,
        "only_images": only_imgs
    }
    logger.info(f"Sending request to {url} with data: {data}")
    response = requests.post(url, json=data)

    if response.status_code == 200:
        logger.info("Request successful.")
        return response.json()
    else:
        logger.error(f"Request failed with status code {response.status_code}: {response.text}")
        # raise Exception(f"Failed to query knowledge base: {response.text}")
        return {"msg": "error"}
    

class ZhipuAIClient(OpenAIClient):
    def __init__(
            self,
            model_name,
            api_key,
            system_prompt=INITIAL_SYSTEM_PROMPT,
            temperature=1.0,
            top_p=1.0,
            user_name="",
    ) -> None:
        super().__init__(
            api_key = api_key,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
            # user=user_name,
        )
        self.api_key = api_key
        self.need_api_key = True
        self._refresh_header()
        self.client = None
        # self.user_name = user_name
        logger.info(f"user name: {user_name}")

    def get_answer_stream_iter(self):
        if not self.api_key:
            raise ValueError("API key is not set")
        response = self._get_response(stream=True)
        if response is not None:
            stream_iter = decode_chat_response(response)
            partial_text = ""
            for chunk in stream_iter:
                partial_text += chunk
                yield partial_text
        else:
            yield STANDARD_ERROR_MSG + GENERAL_ERROR_MSG

    # def get_answer_at_once(self):
    #     if not self.api_key:
    #         raise ValueError("API key is not set")
    #     response = self._get_response()
    #     response = json.loads(response.text)
    #     content = response["choices"][0]["message"]["content"]
    #     total_token_count = response["usage"]["total_tokens"]
    #     return content, total_token_count


    @shared.state.switching_api_key  # 在不开启多账号模式的时候，这个装饰器不会起作用
    def _get_response(self, stream=False):
        zhipuai_api_key = self.api_key
        system_prompt = self.system_prompt
        history = self.history
        # logger.debug(f"{history}")
        # headers = {
        #     "Authorization": f"Bearer {zhipuai_api_key}",
        #     "Content-Type": "application/json",
        # }

        if system_prompt is not None:
            history = [construct_system(system_prompt), *history]

        payload = {
            "model": self.model_name,
            "messages": history,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n_choices,
            "stream": stream,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

        if self.max_generation_token is not None:
            payload["max_tokens"] = self.max_generation_token
        if self.stop_sequence is not None:
            payload["stop"] = self.stop_sequence
        if self.logit_bias is not None:
            payload["logit_bias"] = self.logit_bias
        if self.user_identifier:
            payload["user"] = self.user_identifier

        if stream:
            timeout = TIMEOUT_STREAMING
        else:
            timeout = TIMEOUT_ALL

        # 如果有自定义的api-host，使用自定义host发送请求，否则使用默认设置发送请求
        # if shared.state.chat_completion_url != CHAT_COMPLETION_URL:
            # logger.debug(f"使用自定义API URL: {shared.state.chat_completion_url}")

        # with config.retrieve_proxy():
        #     try:
        #         response = requests.post(
        #             shared.state.chat_completion_url,
        #             headers=headers,
        #             json=payload,
        #             stream=stream,
        #             timeout=timeout,
        #         )
        #     except Exception as e:
        #         logger.error(f"Error: {e}")
        #         response = None
        # return response

        if self.client is None:
            self.client = ZhipuAI(api_key = zhipuai_api_key)

        response = self.client.chat.completions.create(
                model=self.model_name,
                # model="glm-3-turbo",
                messages=history,
                temperature=self.temperature,
                top_p= self.top_p,
                stream= stream,
            )

        
        # "n": self.n_choices,
        # "stream": stream,
        # "presence_penalty": self.presence_penalty,
        # "frequency_penalty": self.frequency_penalty,
        print("response in zhipuai ",response, type(response))
        return response
    

    # todo: fix bug
    def billing_info(self):
            status_text = "获取API使用情况失败，未更新ZhipuAI代价代码。"
            return status_text
            # try:
            #     curr_time = datetime.datetime.now()
            #     last_day_of_month = get_last_day_of_month(
            #         curr_time).strftime("%Y-%m-%d")
            #     first_day_of_month = curr_time.replace(day=1).strftime("%Y-%m-%d")
            #     usage_url = f"{shared.state.usage_api_url}?start_date={first_day_of_month}&end_date={last_day_of_month}"
            #     try:
            #         usage_data = self._get_billing_data(usage_url)
            #     except Exception as e:
            #         logger.warning(f"获取API使用情况失败:" + str(e))
            #         return i18n("**获取API使用情况失败**")
            #     rounded_usage = "{:.5f}".format(usage_data["total_usage"] / 100)
            #     return i18n("**本月使用金额** ") + f"\u3000 ${rounded_usage}"
            # except requests.exceptions.ConnectTimeout:
            #     status_text = (
            #             STANDARD_ERROR_MSG + CONNECTION_TIMEOUT_MSG + ERROR_RETRIEVE_MSG
            #     )
            #     return status_text
            # except requests.exceptions.ReadTimeout:
            #     status_text = STANDARD_ERROR_MSG + READ_TIMEOUT_MSG + ERROR_RETRIEVE_MSG
            #     return status_text
            # except Exception as e:
            #     import traceback
            #     traceback.print_exc()
            #     logger.error(i18n("获取API使用情况失败:") + str(e))
            #     return STANDARD_ERROR_MSG + ERROR_RETRIEVE_MSG


    # def predict(
    #         self,
    #         inputs,
    #         chatbot,
    #         stream=False,
    #         use_websearch=False,
    #         files=None,
    #         reply_language="中文",
    #         should_check_token_count=True,
    # ):
    #     logger.info(f"inputs: {inputs}")
    #     print(f"inputs: {inputs}")

    #     # status_text = "Hello, World! this is zhipu fake output in predict"

    #     response = query_knowledge_base(query=inputs, kb_name="lic")
    #     # response = query_knowledge_base(query=inputs, kb_name="lic", only_imgs=True)["img_urls"]
    #     print("type",type(response),response)
    #     answer = response.get("answer", None)
    #     if answer is None:
    #         # 调用父类，用chat模型回答。
    #         return super().predict(inputs, chatbot, stream, use_websearch, files, reply_language, should_check_token_count)

    #     chatbot = chatbot + [(inputs, "")]
    #     for char in answer:
    #         prev_resp = chatbot[-1][1]
    #         chatbot[-1] = (inputs, prev_resp+char)
    #         yield chatbot, answer


        # yield chatbot + [(inputs, status_text)], status_text
        # for char in status_text:
        #     yield chatbot + [(inputs, char)], char

        # todo: 在这里面用agent做路由：1 chat（or online chat） 2 rag 知识问答 3 多模态问答 4 儿童模式（知识和图片、视频全都幼儿化）