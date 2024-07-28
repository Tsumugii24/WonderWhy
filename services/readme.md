
安装与配置环境
1. 安装环境
为了确保系统环境的一致性，我们推荐使用conda来创建和管理Python虚拟环境。请按照以下步骤操作：

使用conda创建名为rag_backend的新环境，并指定Python版本为3.10：


```bash
conda create -n rag_backend python=3.10
```
激活rag_backend环境：

```bash
conda activate rag_backend
```
安装项目依赖，通过pip安装services/requirements.txt文件中列出的所有必需包：

```bash
pip install -r services/requirements.txt
```

2. 启动服务
启动后端服务，以供前端调用和处理业务逻辑：

```bash
python services/run.py
```

3. 建立向量索引
使用zhipu的glm-4模型，对processed_data目录下的所有Markdown文件进行向量化处理。以下是具体步骤：

3.1 向量化处理
执行以下命令，将Markdown文件向量化并存储到指定的知识库中：

```bash
python clients/ingestion_cli.py -d processed_data/ -k lic --save_imgs
```

3.2 测试向量化效果
通过以下命令测试向量化后的问答效果：

测试RAG问答功能：

```bash
python clients/rag_cli.py -q "有哪些海怪" -k lic
```
测试图片路径回答功能：

```bash
python clients/rag_cli.py -q "哪有海怪" -k lic --only_images
```

4. 启动WonderWhy服务
切换到wonderwhy项目环境，根据需要修改配置文件，并启动服务。注意，如果将default_model设置为rag，则可以在默认回答中直接调用RAG模型：

```bash
python main.py
```

5. 启动Sunny服务
执行以下命令启动sunny服务，并输入clientid以将8010端口映射到指定的URL地址：

```bash
python sunny.py
```

然后，在浏览器中访问：
[WonderWhy Service](http://wonderwhy.v6.idcfengye.com)
或者0.0.0.0:8010