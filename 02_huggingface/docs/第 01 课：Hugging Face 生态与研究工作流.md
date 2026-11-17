# Hugging Face 系统课程 · 第一阶段

正式开课

学习方向：AI Research / 模型训练 / 实验复现

根据你的技术基础，我们采用 研究型开发路线（Research Track），而不是普通的 AI 应用入门路线。

你的学习环境

Python 水平

中级 · 可独立开发

操作系统

Linux

计算资源

NVIDIA GPU

最终方向

模型训练与 AI 研究

这意味着我们的课程将有三个明显的变化：

1. 跳过 Python 基础语法，直接学习 Hugging Face SDK、模型仓库、版本管理、GPU 环境和研究工作流。
2. 强化底层原理，不满足于使用 `pipeline()`，后续会深入 Transformer 架构、Tokenizer、梯度计算、模型参数、训练器和分布式训练。
3. 强调实验可复现性，从第一阶段开始记录模型版本、数据集版本、随机种子、软件环境和 GPU 信息。

## 第一阶段的课程安排

01

Hugging Face 生态与研究工作流

理解 Hub、模型、数据集、仓库结构

当前

02

Linux + NVIDIA GPU 开发环境

PyTorch、CUDA、驱动、虚拟环境

03

Hugging Face Hub 与身份认证

Token、权限、CLI、缓存

04

模型仓库与版本控制

Git、Git LFS、Xet、Revision、Commit

05

huggingface_hub Python SDK

搜索、下载、上传、API 自动化

06

模型文件与配置解析

config.json、权重、Tokenizer、Safetensors

07

Datasets 与研究数据管理

Dataset 仓库、数据格式、版本固定

08

研究规范与阶段实战

实验记录、可复现性、模型仓库发布

预计 10～16 小时完成第一阶段。每课通过实践检查后再进入下一课。

第一阶段的最终成果是建立一个 可复现的 Hugging Face 模型研究工作区：能固定模型版本、自动下载研究材料、验证 GPU 环境，并生成实验元数据。

# 第 01 课：Hugging Face 生态与研究工作流

难度：基础 → 中级

&#x20;60～90 分钟

## 1.1 Hugging Face 的核心设计

对于 AI 研究人员，Hugging Face 最重要的价值不仅是提供预训练模型，而是让模型、数据集、代码和实验结果可以被共享、追踪和复现。

Hugging Face Hub

统一管理模型、数据集、应用与版本

Models

模型权重与配置

Datasets

训练与评测数据

Spaces

应用与实验演示

研究与训练工具链

Transformers / Datasets

PEFT / TRL

Accelerate

Evaluate / Gradio

我们需要区分三个概念：

Hub 是平台，Transformers 是库，模型是资产。

例如：

- Hub：存储模型及其历史版本。
- Transformers：负责加载模型、执行推理、参与训练。
- 模型仓库：保存权重、配置、Tokenizer 和模型说明。

这三者可以独立存在，但组合使用能够形成完整的研究工作流。

## 1.2 认识模型仓库

打开这个公开模型：

[DistilBERT 模型仓库](https://huggingface.co/distilbert/distilbert-base-uncased?utm_source=chatgpt.com)

你会看到类似以下的文件：

```
distilbert-base-uncased/
├── README.md
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```

这是用于教学的简化结构，实际仓库文件可能更多。

每种文件都有不同作用：

| 文件                      | 作用             |
| ----------------------- | -------------- |
| `README.md`             | 模型卡，说明用途、训练与限制 |
| `config.json`           | 模型架构与超参数配置     |
| `model.safetensors`     | 模型权重           |
| `tokenizer.json`        | 分词器定义          |
| `tokenizer_config.json` | 分词器相关设置        |
| `vocab.txt`             | 词表             |

这里有一个非常重要的研究概念：

模型架构 ≠ 模型权重 ≠ Tokenizer。

模型架构定义神经网络如何计算；权重是训练得到的参数；Tokenizer 决定文本如何转换成 Token ID。

例如，同一种 Transformer 架构可以加载不同训练得到的权重，而使用错误的 Tokenizer 可能导致输入表示不匹配，甚至直接报错。

## 1.3 模型仓库的版本控制

Hugging Face 的模型、数据集和 Spaces 仓库支持 Git 风格的版本控制。大型模型文件使用专门的存储机制，当前官方文档重点介绍 Xet；它提供分块去重等能力，Git LFS 仍有兼容支持。

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face



研究中你需要掌握三个标识：

| 标识         | 示例     | 特点              |
| ---------- | ------ | --------------- |
| Branch     | `main` | 可以随提交变化         |
| Tag        | `v1.0` | 通常用于发布版本，但仍可被移动 |
| Commit SHA | 一串完整哈希 | 精确定位某次提交        |

假设一篇论文只写：

```
model_name = "distilbert/distilbert-base-uncased"
```

这只能说明使用了哪个模型仓库，不能严格保证未来下载的文件与实验当天完全相同。

更严谨的做法：

```
model_name = "distilbert/distilbert-base-uncased"revision = "实际查询得到的完整Commit SHA"
```

之后通过 `revision` 固定模型版本。数据集也应采用同样的方法。

这是我们之后复现论文、比较训练实验和发布研究结果的基础。

## 1.4 什么是研究级可复现性？

假设你训练了一个分类模型，准确率达到 92%。

三个月后重新运行，结果却变成 89%。

可能的原因包括：

- 训练数据发生变化。
- 模型初始权重发生变化。
- PyTorch 或 Transformers 版本不同。
- 随机种子没有固定。
- CUDA 算法、硬件或数值精度不同。
- 训练超参数没有完整保存。

因此，我们会在每个研究项目中建立一个实验记录：

```
experiment/
├── configs/
│   └── train.yaml
├── src/
│   └── train.py
├── results/
│   └── metrics.json
├── metadata/
│   └── environment.json
└── README.md
```

需要注意：固定随机种子和依赖版本能够提高复现性，但不保证跨 GPU、CUDA 版本和不同算法实现时达到逐位相同的结果。

# 第 01 课实操：搭建研究工作区

本次实践只需要 CPU 和网络。NVIDIA GPU 检测会在下一课进行。

## 步骤一：建立项目目录

在 Linux 终端执行：

```
mkdir -p ~/hf-research
cd ~/hf-research

mkdir -p src configs results metadata
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --upgrade huggingface_hub
```

验证：

```
hf --help
```

当前官方 CLI 使用 `hf` 命令，支持下载、认证、缓存管理等操作。

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face



## 步骤二：查询模型元数据

创建 `../backup/src`：

```
import jsonfrom pathlib import Pathfrom huggingface_hub import HfApiMODEL_ID = "distilbert/distilbert-base-uncased"api = HfApi()info = api.model_info(    repo_id=MODEL_ID,    files_metadata=True,)result = {    "model_id": info.id,    "commit_sha": info.sha,    "pipeline_tag": info.pipeline_tag,    "library_name": info.library_name,    "files": [        {            "name": f.rfilename,            "size": f.size,        }        for f in info.siblings or []    ],}output = Path("metadata/model_info.json")output.parent.mkdir(parents=True, exist_ok=True)output.write_text(    json.dumps(result, indent=2, ensure_ascii=False),    encoding="utf-8",)print("Model:", result["model_id"])print("Revision:", result["commit_sha"])print("Files:", len(result["files"]))print("Saved:", output)
```

执行：

```
python src/inspect_model.py
```

这个程序使用 `HfApi` 从 Hub 查询模型信息，并将模型的完整 Commit SHA 和文件清单写入 JSON。

它不会下载完整模型权重，因此适合作为研究的第一步。

## 步骤三：下载并固定模型配置

创建 `../backup/src`：

```
import jsonfrom pathlib import Pathfrom huggingface_hub import hf_hub_downloadmetadata = json.loads(    Path("metadata/model_info.json").read_text(        encoding="utf-8"    ))file_path = hf_hub_download(    repo_id=metadata["model_id"],    filename="config.json",    revision=metadata["commit_sha"],)config = json.loads(    Path(file_path).read_text(encoding="utf-8"))print("Downloaded file:", file_path)print("Model type:", config.get("model_type"))print("Hidden dimension:", config.get("dim"))print("Number of layers:", config.get("n_layers"))print("Attention heads:", config.get("n_heads"))
```

运行：

```
python src/download_config.py
```

你应该能看到模型架构信息，其中 DistilBERT 的典型配置为：

| 参数                 | 数值  |
| ------------------ | --- |
| Hidden dimension   | 768 |
| Transformer layers | 6   |
| Attention heads    | 12  |

这个实验体现了一个完整的研究流程：

获取元数据 → 记录 Commit SHA → 按指定版本下载 → 解析配置。

官方下载文档：[Download files from the Hub](https://huggingface.co/docs/huggingface_hub/main/guides/download?utm_source=chatgpt.com)。

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face



## 第 01 课练习与验收

请完成以下 5 项任务。前 3 项是必做，第 4～5 项是研究型进阶练习。

练习清单

0/5 完成

完成研究环境初始化

创建 hf-research 项目和 Python 虚拟环境

运行 inspect_model.py

生成 metadata/model_info.json

运行 download_config.py

成功读取 DistilBERT 的模型架构参数

分析模型版本

解释 main 分支与 Commit SHA 在实验复现中的区别

研究模型结构

解释 Hidden Size、Layer、Attention Head 各自的含义

提交学习进度并进行验收 完成前三项必做练习后，可以提交进度。

## 研究思考题

尝试独立回答下面三个问题，不需要查找标准答案：

1. 为什么 `config.json` 不能代替 `model.safetensors`？
2. 如果两位研究者使用同一个模型仓库，但没有固定 Commit SHA，为什么可能得到不同的实验结果？
3. 一个 768 维、12 个 Attention Head 的 Transformer，通常每个 Head 的维度是多少？这个结果依赖什么架构假设？

这些问题会帮助我判断你对模型结构和研究可复现性的理解程度。

## 下一课预告：Linux + NVIDIA GPU 训练环境

第二课将深入以下技术：

- NVIDIA Driver、CUDA Driver API、CUDA Runtime、CUDA Toolkit 的区别。
- `nvidia-smi` 与 `nvcc` 分别检查什么。
- PyTorch CUDA 版本和系统 CUDA 版本的关系。
- GPU Tensor 运算、显存分配、FP32、FP16、BF16。
- GPU 性能测试与训练环境诊断。

你可以提前执行下面三条命令，保留输出：

```
nvidia-smi
python3 --version
cat /etc/os-release
```

下一步： 先完成第 01 课的代码实践，把终端输出或遇到的错误发给我。我会根据实际结果检查环境和代码，完成知识验收，然后正式进入第 02 课。

这样我们能够确保每一阶段不仅学过，而且真正掌握。