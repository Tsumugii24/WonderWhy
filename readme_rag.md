## 1  环境

```bash
$ conda create -n wonderwhy python==3.10 
$ conda activate wonderwhy
```


```bash
$ git clone https://github.com/Tsumugii24/WonderWhy && cd WonderWhy
$ pip install -r requirements.txt
```


下载向量模型

```bash
apt-get update 
apt-get install git-lfs curl aria2

wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc && source ~/.bashrc
echo 'export PATH="$PATH:$PWD"' >> ~/.bashrc && echo 'alias hfd="$PWD/hfd.sh"' >> ~/.bashrc && source ~/.bashrc

hfd BAAI/bge-m3 --local-dir models/bge
```


修改config.json
```

{
    "openai_api_key": "ab13c3a2xxxxxxxxxxx", // ZhipuAI api key
    "language": "auto",
    "local_embedding": true, 
    "hf_emb_model_name": "/home/jhx/Projects/WonderWhy/bge-m3/", # local embed model
    "hide_history_when_not_logged_in": true,
    "default_model": "glm-3-turbo",
    "multi_api_key": false,
    // "server_name": "127.0.0.1",
    "server_name": "0.0.0.0",
    "server_port": 8010,
    "autobrowser": false,
    "share": true,
    "websearch_engine": "duckduckgo"
}
```


部署：

```shell
python main.py
python python sunny.py  --clientid=0118xxxxxxx
```


v0.1.1 新增zhipuai api

- src/zhipu_client.py # 智谱ai调用代码
- base_model添加zhipuai模型的注册
- presets增加可选的模型
- models增加调用zhipuai模型client
- 修改config.json

To fix: 
1 找不到修改stream的按钮
2 文件上传后，重置对话轮，然后会error说没有传文件。