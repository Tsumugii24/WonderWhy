
import base64
import datetime
import json
import os
from io import BytesIO
import requests
import time

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
    

class RAGClient(OpenAIClient):
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

    # todo
    def get_answer_stream_iter(self):
        # if not self.api_key:
        #     raise ValueError("API key is not set")
        response = self._get_response(stream=True)
        if response is not None:
            stream_iter = response
            partial_text = ""
            for chunk in stream_iter:
                partial_text += chunk
                time.sleep(0.005)
                yield partial_text
        else:
            yield STANDARD_ERROR_MSG + GENERAL_ERROR_MSG

    # todo
    def get_answer_at_once(self):
        content = self._get_response()
        total_token_count = len(content)
        return content, total_token_count
    

    # @shared.state.switching_api_key  # 在不开启多账号模式的时候，这个装饰器不会起作用
    def _get_response(self, stream=False):
        system_prompt = self.system_prompt
        history = self.history
        if system_prompt is not None:
            history = [construct_system(system_prompt), *history]

        prompt = history[-1]["content"]
        response = query_knowledge_base(query=prompt, kb_name="lic", only_imgs=False)
        answer = response.get("answer", None)
        if answer is None:
            # TODO: CALL GPT
            answer = "这题超纲了哦~"

        return answer
    

    # todo: fix bug
    def billing_info(self):
            status_text = "获取API使用情况失败，未更新ZhipuAI代价代码。"
            return status_text


