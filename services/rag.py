'''
1 RAG的index，使用bge编码，FAISS进行向量存储。
2 RAG服务： chat_kb
'''
import os
import sys
import re
from typing import List, Optional, Any, Dict
from fastapi import APIRouter, HTTPException
from pathlib import Path
from loguru import logger

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_DIR = os.path.join(ROOT_DIR, "vector_store")


rag_router = APIRouter()


from tqdm import tqdm

from pathlib import Path
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import Settings
import faiss
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

from pydantic import BaseModel

KB_ENGINES: Dict[str, Any] = {}
OLLAMA_URL="http://localhost:11434"
llm = Ollama(model="qwen", base_url=OLLAMA_URL, temperature=0,  request_timeout=120)
embed_model = HuggingFaceEmbedding(
    model_name = "/home/jhx/Projects/pretrained_models/bge-m3/",
    cache_folder="./",
    embed_batch_size=64,
)

Settings.llm = llm
Settings.embed_model = embed_model


class IngestionRequest(BaseModel):
    document_dir: str
    kb_name: str

class ChatRequest(BaseModel):
    query: str
    kb_name: str

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
    

def find_markdown_files(directory, recursive=True):
    """
    Find all Markdown files in the specified directory.

    Args:
        directory (str or Path): The directory to search in.
        recursive (bool): Whether to search recursively in subdirectories.

    Returns:
        list: A list of paths to the Markdown files.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"The directory {directory} does not exist or is not a directory.")
    
    markdown_files = []
    if recursive:
        for file_path in directory.rglob('*.md'):
            markdown_files.append(file_path)
    else:
        for file_path in directory.glob('*.md'):
            markdown_files.append(file_path)
    
    return markdown_files

def vectorize_and_store(files, kb_name):
    """
    Placeholder function for vectorizing a batch of documents and storing them in the database.

    Args:
        files (list of Path): The paths to the documents.
        kb_name (str): The name of the knowledge base.

    Returns:
        dict: A summary of the operation.
    """
    # todo: llama_index store

    # init 
    global embed_model
    kb_path = Path(VECTOR_STORE_DIR) / kb_name

    Settings.embed_model = embed_model
    d = 1024
    faiss_index = faiss.IndexFlatL2(d)
    chunk_size = 1024
    zh_node_parser = LangchainNodeParser(ChineseRecursiveTextSplitter(chunk_size = chunk_size, chunk_overlap= 50))

    # load nodes
    all_nodes = []
    for file in tqdm(files):
        logger.info(f"Processing parse file: {os.path.basename(file)}...")
        documents = FlatReader().load_data(Path(file))
        parser = MarkdownNodeParser() # todo: use langchain split
        md_nodes = parser.get_nodes_from_documents(documents)

        nodes = []
        for node in md_nodes:
            text = node.text
            node_id = node.node_id
            if len(text) <= chunk_size:
                nodes.append(node)
            else:
                doc = Document(text = text, id_ = node_id)
                splitted_nodes = zh_node_parser.get_nodes_from_documents([doc]) # 两两之前有prev和next，开头无prev 结尾无prev
                if len(nodes) != 0 and len(splitted_nodes) >0:
                    prev_node = nodes[-1]
                    nodes[-1].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id = splitted_nodes[0].node_id)
                    splitted_nodes[0].relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=prev_node.node_id)

                if len(splitted_nodes) > 0:
                    nodes.extend(splitted_nodes) # 防止空节点
        all_nodes.extend(nodes)

    logger.info(f"all_nodes: {len(all_nodes)}")
    # index
    logger.info(f"Building index ing...")
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(all_nodes, storage_context=storage_context, embed_model=embed_model, show_progress=True) # store_nodes_override似乎是存储nodes，但是没起到作用

    # storage happen here
    logger.info(f"Storing {len(files)} files in knowledge base {str(kb_name)}.")
    index.storage_context.persist(persist_dir = kb_path)
    
    return {
        "processed_files": [str(file) for file in files],
        "skipped_files": []
    }

@rag_router.post("/ingestion")
async def vectorize_document(request:IngestionRequest):
    """
    Vectorize and store all Markdown files in the specified directory.

    Args:
        document_dir (str): The path to the directory containing the documents.
        kb_name (str): The name of the knowledge base.

    Returns:
        dict: A message indicating the result of the operation.
    """
    document_dir = request.document_dir
    kb_name = request.kb_name

    logger.info(f"document_dir: {document_dir}")
    logger.info(f"kb_name: {kb_name}")

    # 检查知识库路径及其文件是否存在
    kb_required_files = [
        'default__vector_store.json',
        'docstore.json',
        'graph_store.json',
        'image__vector_store.json',
        'index_store.json'
    ]
    
    try:
        kb_path = Path(VECTOR_STORE_DIR) / kb_name
        # Create knowledge base directory if it does not exist
        os.makedirs(kb_path, exist_ok=True)

        # todo: check kb_name
        
        if kb_path.exists() and all((kb_path / file).exists() for file in kb_required_files):
            return {
                "message": f"Knowledge base {kb_name} already exists with all required files, skipping vectorization.",
                "result": {
                    "processed_files": [],
                    "skipped_files": [str(file) for file in find_markdown_files(document_dir, recursive=True)]
                }
            }
        
        # vectorization and storage happen here
        markdown_files = find_markdown_files(document_dir, recursive=True)
        logger.info(f"Vectorizing and storing {len(markdown_files)} files in knowledge base {kb_name}.")

        result = vectorize_and_store(markdown_files, kb_name)
        return {"message": "Documents processed.", "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def initialize_query_engine(kb_name: str) -> Any:
    """初始化查询引擎"""
    kb_path = Path(VECTOR_STORE_DIR) / kb_name
    try:
        vec_store = FaissVectorStore.from_persist_dir(kb_path)
        storage_context = StorageContext.from_defaults(vector_store=vec_store, persist_dir=kb_path)
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine(similarity_top_k=3)
        KB_ENGINES[kb_name] = query_engine
        return query_engine
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize query engine: {str(e)}")


@rag_router.post("/chat_kb")
async def generate_answer(request: ChatRequest):
    query = request.query
    kb_name = request.kb_name

    if kb_name not in KB_ENGINES:
        # 初始化查询引擎
        initialize_query_engine(kb_name)

    query_engine = KB_ENGINES.get(kb_name)
    
    try:
        answer = query_engine.query(query)
        return {"query": query, "answer": str(answer)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")