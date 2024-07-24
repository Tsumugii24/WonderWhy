
## 1 开启服务

python services/run.py 

## 2 建立向量索引

用的是openai的default embed模型：text-embedding-ada-002
python clients/ingestion_cli.py -d processed_data/ -k lic --save_imgs

## 3 开启wonderwhy

