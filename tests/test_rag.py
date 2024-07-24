from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


OLLAMA_URL="http://localhost:11434"
llm = Ollama(model="qwen", base_url=OLLAMA_URL, temperature=0,  request_timeout=120)
embed_model = HuggingFaceEmbedding(
    model_name = "/home/jhx/Projects/pretrained_models/bge-m3/",
    cache_folder="./",
    embed_batch_size=64,
)

Settings.llm = llm
Settings.embed_model = embed_model


vec_store_path = "./storage/" # ocean
vec_store_path = "vector_store/lic"
# vec_store_path = "vector_store/lic_imgs"

vec_store = FaissVectorStore.from_persist_dir(vec_store_path)
storage_context = StorageContext.from_defaults(vector_store= vec_store, persist_dir=vec_store_path)

index = load_index_from_storage(storage_context)

###### 重新获取nodes ######
# nodes = index.docstore.docs
# print("index nodes _docstore")
# # print(len(nodes), type(nodes[0], nodes[0])) # <llama_index.core.storage.docstore.simple_docstore.SimpleDocumentStore
# print(len(nodes)) # <llama_index.core.storage.docstore.simple_docstore.SimpleDocumentStore
# for node_id, node in nodes.items():
#     print(type(node),node_id, node)
##########################
retriever = index.as_retriever(similarity_top_k=2)
from llama_index.core.schema import NodeWithScore

nodes = retriever.retrieve("海洋水怪有哪些")
print(len(nodes))

for node in nodes:
    print(type(node), node.node, node.score) # llama_index.core.schema.NodeWithScore

# print(nodes )
query_engine = index.as_query_engine(
        similarity_top_k = 3
)

resp = query_engine.query("海洋怪兽有哪些")

print(resp)


# query_engine = index.as_query_engine(
#     similarity_top_k=2,
#     llm=llm,
#     # 目标键默认设置为 window，以匹配 node_parser 的默认设置
#     node_postprocessors=[
#         MetadataReplacementPostProcessor(target_metadata_key="window"),
#         cohere_rerank
#     ],
# )