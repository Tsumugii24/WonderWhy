import os
import re
import json
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter


from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.bridge.pydantic import PrivateAttr


from zhipuai import ZhipuAI


def set_api_key_from_json(json_file_path):
    try:
        # 打开并读取JSON文件
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        # 从JSON数据中获取api_key
        api_key= data.get('openai_api_key','sk-')
        api_base= data.get('openai_api_base','https://apikeyplus.com/v1')
        model_name= data.get('default_model','gpt-3.5-turbo')
        zhipuai_api_key= data.get('zhipuai_api_key','gpt-3.5-turbo')


        # 设置环境变量
        os.environ['zhipuai_api_key'] = zhipuai_api_key
        os.environ['openai_api_key'] = api_key
        os.environ['openai_api_base'] = api_base
        os.environ['default_model'] = model_name
    except FileNotFoundError:
        print(f"The file {json_file_path} does not exist.")
    except json.JSONDecodeError:
        print("Error decoding the JSON file.")
    except Exception as e:
        print(f"An error occurred: {e}")



class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """Recursive text splitter for Chinese text.
    copy from: https://github.com/chatchat-space/Langchain-Chatchat/tree/master
    """

    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s"
        ]
        self._is_separator_regex = is_separator_regex

    @staticmethod
    def _split_text_with_regex_from_end(
            text: str, separator: str, keep_separator: bool
    ) -> List[str]:
        # Now that we have the separator, split the text
        if separator:
            if keep_separator:
                # The parentheses in the pattern keep the delimiters in the result.
                _splits = re.split(f"({separator})", text)
                splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
                if len(_splits) % 2 == 1:
                    splits += _splits[-1:]
            else:
                splits = re.split(separator, text)
        else:
            splits = list(text)
        return [s for s in splits if s != ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = self._split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip() != ""]





class ZhipuAIEmbeddings(BaseEmbedding):
    _client: ZhipuAI = PrivateAttr()
    _api_key: str = PrivateAttr()
    
    def __init__(
        self,
        model_name: str = "embedding-2",
        api_key: str = "",
        **kwargs: Any,
    ) -> None:
        # self.model_name = model_name
        self._client = None
        self._api_key = api_key

        super().__init__(model_name=model_name, **kwargs)


    def _get_client(self) -> ZhipuAI:
        if self._client is None:
            self._client = ZhipuAI(api_key= self._api_key)
        return self._client
    
    @classmethod
    def class_name(cls) -> str:
        return "ZhipuAIEmbeddings"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        response = self._get_client().embeddings.create(
                        model=self.model_name, 
                        input=query
                        )
        embedding = response.data[0].embedding

        return embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        response = self._get_client().embeddings.create(
                        model=self.model_name, 
                        input=text
                        )
        embedding = response.data[0].embedding

        return embedding
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self._get_client().embeddings.create(
                        model=self.model_name, 
                        input=texts
                        )
        embeddings = [data.embedding for data in response.data]

        return embeddings
    

class ZhipuAILLM(CustomLLM):
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "glm-4"

    _client: ZhipuAI = PrivateAttr()
    _api_key: str = PrivateAttr()

    def __init__(
        self,
        model_name: str = "glm-4",
        api_key: str = "",
        **kwargs: Any,
    ) -> None:
        # self.model_name = model_name
        self._client = None
        self._api_key = api_key

        super().__init__(model_name=model_name, **kwargs)

    def _get_client(self) -> ZhipuAI:
        if self._client is None:
            self._client = ZhipuAI(api_key= self._api_key)
        return self._client

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages=[
            {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
            {"role": "user", "content": prompt},
        ]
    
        response = self._get_client().chat.completions.create(
            model = self.model_name,
            messages= messages,
            stream=False
        )
        content = response.choices[0].message.content
        return CompletionResponse(text= content)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        messages=[
            {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
            {"role": "user", "content": prompt},
        ]
    
        response = self._get_client().chat.completions.create(
            model = self.model_name,
            messages= messages,
            stream=True
        )
        resp = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            resp += delta
            yield CompletionResponse(text = resp, delta=delta)