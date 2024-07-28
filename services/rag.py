'''
1 RAG的index，使用bge编码，FAISS进行向量存储。
2 RAG服务： chat_kb。
'''
import os
import sys
import re
from typing import List, Optional, Any, Dict
from pathlib import Path
__dir__=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__,"../")))

from loguru import logger
from fastapi import APIRouter, HTTPException
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings


from tqdm import tqdm

import faiss
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
from llama_index.core.node_parser import LangchainNodeParser
from pydantic import BaseModel
from utils import ChineseRecursiveTextSplitter, ZhipuAIEmbeddings, ZhipuAILLM


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_DIR = os.path.join(ROOT_DIR, "vector_store")
API_KEY = os.getenv("openai_api_key")
API_BASE = os.getenv("openai_api_base")
MODEL_NAME = os.getenv("default_model", "gpt-3.5-turbo")
ZHIPUAI_API_KEY = os.getenv("zhipuai_api_key")


rag_router = APIRouter()
KB_ENGINES: Dict[str, Any] = {}


EMBED_DIM = 1024
# EMBED_DIM = 1536
CHUNK_SIZE = 1024
EMBED_BSZ = 2



# todo: zhipu llm


## LLM
# OLLAMA_URL="http://localhost:11434"
# llm = Ollama(model="qwen", base_url=OLLAMA_URL, temperature=0,  request_timeout=120)
# llm = OpenAI(model=MODEL_NAME, api_key=API_KEY, api_base=API_BASE)

# from llama_index.legacy.llms import OpenAILike as OpenAI
# llm = OpenAI(
#     api_key=ZHIPUAI_API_KEY,
#     model="glm-4",
#     api_base="https://open.bigmodel.cn/api/paas/v4/",
#     is_chat_model=True,
# )
llm = ZhipuAILLM(model_name="glm-4", api_key=ZHIPUAI_API_KEY)


## EMBED MODEL
embed_model = ZhipuAIEmbeddings(embed_batch_size=EMBED_BSZ,api_key=ZHIPUAI_API_KEY)
# embed_model = HuggingFaceEmbedding(
#     # model_name = "/path/to/bge-m3/",
#     model_name = "/home/jhx/Projects/pretrained_models/bge-m3/",
#     cache_folder="./",
#     embed_batch_size=EMBED_BSZ,
# )
# embed_model = OpenAIEmbedding(embed_batch_size=EMBED_BSZ,api_key=API_KEY, api_base=API_BASE)



Settings.llm = llm
Settings.embed_model = embed_model


class IngestionRequest(BaseModel):
    document_dir: str
    kb_name: str
    save_images: bool = False  # 是否保存图片索引

class ChatRequest(BaseModel):
    query: str
    kb_name: str
    topk: int = 3  
    only_images: bool = False  # 是否仅仅检索图片


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

def extract_images_and_descriptions(text):
    """从 Markdown 文本中提取所有图片路径和描述"""
    # 正则表达式匹配图片路径
    image_pattern = r'!\[.*?\]\((.*?)\)'
    image_matches = re.finditer(image_pattern, text)
    
    images = []
    index = 0
    while index < len(text):
        # 查找下一个图片
        image_match = next(image_matches, None)
        if image_match is None:
            break
        
        image_path = image_match.group(1)
        start_pos = image_match.start()
        
        # 查找图片描述
        description_start = text.find('\n', start_pos) + 1
        if description_start == 0:
            description_start = start_pos
        
        # 确保描述与图片路径分开，且不包含其他图片路径
        description_end = text.find('\n', description_start)
        if description_end == -1:
            description_end = len(text)
        
        description = text[description_start:description_end].strip()
        
        # 确保描述不包含图片路径
        # if re.search(image_pattern, description):
        #     description = ""  # 如果描述中包含图片路径，则设置为空
        
        images.append((image_path, description))
        
        # 更新索引位置
        index = description_end + 1
        
    return images

def vectorize_and_store(files, kb_name, save_images):
    """
    Placeholder function for vectorizing a batch of documents and storing them in the database.

    Args:
        files (list of Path): The paths to the documents.
        kb_name (str): The name of the knowledge base.
        save_images(bool): whether to save images, filepath:  vector_store/{kb_name}_imgs

    Returns:
        dict: A summary of the operation.
    """
    # todo: llama_index store

    # init 
    global embed_model
    kb_path = Path(VECTOR_STORE_DIR) / kb_name
    kb_imgs_path = Path(VECTOR_STORE_DIR) / f"{kb_name}_imgs"

    Settings.embed_model = embed_model
    faiss_index = faiss.IndexFlatL2(EMBED_DIM)
    faiss_index_imgs = faiss.IndexFlatL2(EMBED_DIM)
    zh_node_parser = LangchainNodeParser(ChineseRecursiveTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap= 50))

    # load nodes
    all_nodes = []
    all_img_nodes = []

    for file in tqdm(files):
        documents = FlatReader().load_data(Path(file))
        parser = MarkdownNodeParser() # todo: use langchain split
        md_nodes = parser.get_nodes_from_documents(documents)
        
        if save_images:
            logger.info(f"Processing parse imgs of file: {os.path.basename(file)}...")

            # extract img urls and description
            for node in md_nodes:
                text = node.text
                image_desc_pairs = extract_images_and_descriptions(text)
                for img_url, desc in image_desc_pairs:
                    if desc.strip() != "":
                        url = os.path.join(os.path.dirname(file), img_url)
                        img_node = TextNode(text = desc, metadata={"img_url": url.strip()})
                        all_img_nodes.append(img_node)

        nodes = []
        for node in md_nodes:
            text = node.text
            node_id = node.node_id
            if len(text) <= CHUNK_SIZE:
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
    if save_images:
        logger.info(f"Building images index ing...")
        vector_store_imgs = FaissVectorStore(faiss_index=faiss_index_imgs)
        storage_context_imgs = StorageContext.from_defaults(vector_store=vector_store_imgs)
        index_imgs = VectorStoreIndex(all_img_nodes, storage_context=storage_context_imgs, embed_model=embed_model, show_progress=True) # store_nodes_override似乎是存储nodes，但是没起到作用

        # storage happen here
        logger.info(f"Storing {len(all_img_nodes)} image nodes in knowledge base {str(kb_name)}.")
        index_imgs.storage_context.persist(persist_dir = kb_imgs_path)
    
    
    logger.info(f"Building text index ing...")
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(all_nodes, storage_context=storage_context, embed_model=embed_model, show_progress=True) # store_nodes_override似乎是存储nodes，但是没起到作用

    # storage happen here
    logger.info(f"Storing {len(all_nodes)} text nodes in knowledge base {str(kb_name)}.")
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
    save_images = request.save_images

    logger.info(f"document_dir: {document_dir}")
    logger.info(f"kb_name: {kb_name}")
    logger.info(f"save_images: {save_images}")

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

        # todo: check kb_name  valid function
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

        result = vectorize_and_store(markdown_files, kb_name, save_images)
        return {"message": "Documents processed.", "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def initialize_query_engine(kb_name: str, topk: int = 3) -> Any:
    """初始化查询引擎"""
    kb_path = Path(VECTOR_STORE_DIR) / kb_name
    try:
        vec_store = FaissVectorStore.from_persist_dir(kb_path)
        storage_context = StorageContext.from_defaults(vector_store=vec_store, persist_dir=kb_path)
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine(similarity_top_k=topk)
        retriever = index.as_retriever(similarity_top_k=topk)
        KB_ENGINES[kb_name] = {"query_engine":query_engine, "retriever":retriever}
        return query_engine
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize query engine: {str(e)}")


@rag_router.post("/chat_kb")
async def generate_answer(request: ChatRequest):
    query = request.query
    kb_name = request.kb_name
    topk = request.topk
    only_images = request.only_images

    if only_images:
        kb_name = kb_name + "_imgs" 

    if kb_name not in KB_ENGINES:
        # 初始化查询引擎
        initialize_query_engine(kb_name, topk)

    
    try:
        if only_images:
            retriever = KB_ENGINES.get(kb_name)["retriever"] 
            nodes_with_score = retriever.retrieve(query) 
            ret =  {"query": query, "img_urls":  []}
            for node_with_score in nodes_with_score:
                img_node = node_with_score.node
                img_url = img_node.metadata["img_url"]
                ret["img_urls"].append(img_url)
            return ret

        else:
            try:
                query_engine = KB_ENGINES.get(kb_name)["query_engine"] 
                answer = query_engine.query(query)
            except Exception as e:
                logger.info(f"error : {str(e)}")
                answer = "error"
            return {"query": query, "answer": str(answer)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")