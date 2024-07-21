1 开启本地ollama qwen服务

2 启动services服务
python services/run.py 

3 向量编码
python clients/ingestion_cli.py -d processed_data -k lic

4 启动wonderwhy的服务

模型选择用glm-4(会走rag)