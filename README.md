# NFRA 智能问答系统
## 项目介绍
该项目是一个基于国家金融监督管理总局（National Financial Regulatory Administration，简称 NFRA）政策法规构建的 RAG， 
旨在对 NFRA上的政策法规文件进行智能检索、问答，适用于金融政策、法规检索等场景的智能化应用开发与研究。
### 技术栈
- Python 3.12
- LangChain 0.3.17
- Milvus 2.5.4
- Docker 20.10.24
### 项目结构
- `data/`：存放 NFRA 政策法规文件的目录
- `evaluation/`：存放评估数据集和代码的目录
- `project_source/`：存放本文档需要用到的资源的目录
- `src/`：存放源代码的目录
- `.env`：存放环境变量配置文件
- `application.py`：应用程序入口文件，包含Milvus 向量集合的初始化和问答逻辑
- `config.py`：代码配置文件，包含知识文档路径、分块、嵌入、向量索引与存储等配置
- `docker-compose.yml`：Docker Compose 配置文件，用于启动 Milvus 服务
- `requirements.txt`：项目依赖包列表

### 核心代码流程图
初始化 milvus，执行的是：application.load_data_milvus() 方法：
![core_flow](/project_source/initMilvus.png)

问答流程图：
![core_flow](/project_source/qaFlow.png)

## 项目搭建与运行
### 搭建说明
适用于：Windows 本地。
### 搭建与运行
1. 下载项目代码，导入编辑器
2. 配置 Python解释器
3. 安装依赖包，运行以下命令：
    ```bash
    pip install -r requirements.txt
    ```
4. 配置环境变量，在`.env` 文件中填写你的 DeepSeek API Key
5. 安装 Milvus，[在github上](https://github.com/milvus-io/milvus/releases/tag/v2.5.4)，下载并安装 Milvus 2.5.4 版本
6. 安装 Docker 并启动，可参考：[博客](https://blog.csdn.net/QQ1817117243/article/details/139879440?ops_request_misc=%257B%2522request%255Fid%2522%253A%252239eda5b68df6b07564b68f3511c0444a%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=39eda5b68df6b07564b68f3511c0444a&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-139879440-null-null.142^v102^pc_search_result_base5&utm_term=docker%20windows%E5%AE%89%E8%A3%85&spm=1018.2226.3001.4187)
7. 启动环境：
   - （1）运行：docker-compose.yml 文件，启动 Milvus 服务；
   - （2）运行如下命令：（加载 /data 目录下的文档、切分、嵌入、存储到 milvus向量数据库，完成向量初始化）
        ```bash
        python application.py --init
        ```
   - （3）运行如下命令：（通过 shell 方式，启动问答对话）
        ```bash
        python application.py --shell
        ```
8. 问答对话演示：
![core_flow](/project_source/run_result.png)

## 项目延展
- 项目详细介绍可参考[博客](https://blog.csdn.net/quf2zy/article/details/149504959?spm=1011.2415.3001.5331)