<div align="center"><h1>PoetryChat</h1></div>

</div>

<div align="center"><h2>Description</h2></div>

&emsp;&emsp;Powered by Large Language Models, PoetryChat ...



</div>

<div align="center"><h2>Demonstration</h2></div>

![诗趣伴行](./docs/诗趣伴行.png)

&emsp;&emsp;You can easily and directly experience the our demo online on `HuggingFace` now. Click here for Online Experience 👉 [PoetryChat - a Hugging Face Space by jjyaoao](https://huggingface.co/spaces/jjyaoao/poetryChat)

</div>

<div align="center"><h2>Todo</h2></div>

- P1
    - [ ] Export picture "books" for children
    - [ ] Export cards with poet and picture, for recitation
    - [ ] Export map-liked records for parent-child interaction
    - [ ] Test and exsercise for students
- P2
    - [ ] Complete the Gradio Interface and UI design
    - [ ] Add team members brief introduction
    - [ ] Add a gif demonstration
    - [ ] Deploy the demo on HuggingFace
    - [ ] RAG layer
    - [ ] LLM Agent layer
    - [ ] Application layer

</div>


<div>
    <div align="center"><h2>Development Logs</h2></div>

> Remember to update the `README.md` and `requirements.txt` after each commit!

- [2024-07-05] Historic Talk! 



</div>







<div align="center"><h2>Quick Start</h2></div>

<details open>
    <summary><h4>Installation</h4></summary>


&emsp;&emsp;First of all, please make sure that you have already installed `conda` as Python runtime environment. And `miniconda` is strongly recommended.

&emsp;&emsp;1. create a virtual `conda` environment for the demo 😆

```bash
$ conda create -n poetrychat python==3.10 # poetrychat is the name of your environment
$ conda activate poetrychat
```

&emsp;&emsp;2. Install essential `requirements` by run the following command in the `CLI` 😊

```bash
$ git clone https://github.com/Antony-Zhang/PoetryChat && cd PoetryChat && git checkout poetryChat2.0
$ pip install -r requirements.txt
```

<details open>
    <summary><h4>Preparation</h4></summary>


&emsp;&emsp;1. open `config_example.json` and fill your own `API Keys` in the **corresponding place** if you want to use certain LLM, then **rename** the file into `config.json`

```json
// Openai
"openai_api_key": "",
"openai_api_base": "",
// Spark Desk
"sparkdesk_apisecret": "",
"sparkdesk_apikey": "",
"sparkdesk_appid": "",
// Bing
"bing_search_api_key": ""
```

&emsp;&emsp;2. xxx









</div>

<div align="center"><h2>References</h2></div>

1. [Gradio Official Documents](https://www.gradio.app/)
2. [LIC·2024 语言与智能技术竞赛_飞桨大赛-飞桨AI Studio星河社区](https://aistudio.baidu.com/competition/detail/1171/0/introduction)
3. [PoetryChat: 一个面向不同年龄段的交互式LLM古诗学习助手](https://github.com/Antony-Zhang/PoetryChat)





</div>

<div align="center"><h2>Acknowledgements</h2></div>

&emsp;&emsp;***I would like to express my sincere gratitude to my teammates  for their efforts and supports throughout the development of this project. Their expertise and insightful feedback played a crucial role in shaping the direction of the project.***

- [@Tsumugii24](https://github.com/Tsumugii24)

- [@jiaohui](https://github.com/jiaohuix)

- [@jjyaoao](https://github.com/jjyaoao)



</div>

<div align="center"><h2>Contact</h2></div>

Feel free to open GitHub issues or directly send me a mail if you have any questions about this project. 

