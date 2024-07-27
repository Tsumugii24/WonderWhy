<div align="center"><h1>WonderWhy</h1></div>

</div>

<div align="center"><h2>Description</h2></div>

&emsp;&emsp;**WonderWhy** is a cross-platform family education AI application based on large language models (LLMs) and generative AI technologies. It aims to help parents better respond to various questions raised by their children, providing a high-quality educational interaction platform for kids. The goal is to guide children to think independently and cultivate proper values by offering inspiring, artistic, accessible, and engaging solutions.

&emsp;&emsp;Combining with self-media marketing, **WonderWhy** creates professional and in-depth scientific parenting solutions. It leverages high-quality information to break down regional educational barriers, reduce educational information disparities, and accelerate educational equity.



</div>

<div align="center"><h2>Demonstration</h2></div>

![image-20240527204109978](https://cdn.jsdelivr.net/gh/Tsumugii24/Typora-images@main/images/2024%2F05%2F27%2F11a70d1277ce384a55da2cc89ae83bd5-image-20240527204109978-71c0aa.png)

&emsp;&emsp;You can easily and directly experience the our demo online on `HuggingFace` now. Click here for Online Experience 👉 [WonderWhy - a Hugging Face Space by Tsumugii](https://huggingface.co/spaces/Tsumugii/WonderWhy)

</div>

<div align="center"><h2>Todo</h2></div>

- [x] Complete the Gradio Interface and UI design
- [x] Add team members brief introduction
- [x] Add a gif demonstration 👉 https://www.bilibili.com/video/BV1Pt8teWEqJ/
- [ ] Deploy the demo on HuggingFace
- [x] RAG layer
- [x] Agent layer
- [x] Application layer







<div>
    <div align="center"><h2>Development Logs</h2></div>

> Remember to update the `README.md` and `requirements.txt` after each commit!
- [2024-07-05] Historic Talk! 







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
$ git clone https://github.com/Tsumugii24/WonderWhy && cd WonderWhy
$ pip install -r requirements.txt
```

<details open>
    <summary><h4>Preparation</h4></summary>
&emsp;&emsp;Open `config_example.json` and fill your own `API Keys` in the **corresponding place** if you want to use certain LLM, then **rename** the file into `config.json`

```json
// OpenAI
"openai_api_key": "",
"openai_api_base": "",
// Spark Desk
"sparkdesk_apisecret": "",
"sparkdesk_apikey": "",
"sparkdesk_appid": "",
// Bing Search
"bing_search_api_key": ""
// ERNIEBOT configurations is already included in the project
```

<details open>
    <summary><h4>Running</h4></summary>

&emsp;&emsp;`python run main.py`










</div>

<div align="center"><h2>References</h2></div>

1. [Gradio Official Documents](https://www.gradio.app/)
2. [LIC·2024 语言与智能技术竞赛_飞桨大赛-飞桨AI Studio星河社区](https://aistudio.baidu.com/competition/detail/1171/0/introduction)
3. [PoetryChat: 一个面向不同年龄段的交互式LLM古诗学习助手](https://github.com/Antony-Zhang/PoetryChat)





<div align="center"><h2>Acknowledgements</h2></div>

&emsp;&emsp;***I would like to express my sincere gratitude to my teammates  for their efforts and supports throughout the development of this project. Their expertise and insightful feedback played a crucial role in shaping the direction of the project.***

- [@Tsumugii24](https://github.com/Tsumugii24)

- [@jjyaoao](https://github.com/jjyaoao)

- [@jiaohui](https://github.com/jiaohuix)





<div align="center"><h2>Contact</h2></div>

Feel free to open GitHub issues or directly send me a mail if you have any questions about this project ~

