OpenAI 的 **Responses API**（`client.responses.create`）是替代传统 Chat Completions API 的新一代交互接口。它在输入输出格式、系统指令分级、多模态处理、内置工具调用（如网络检索与文件检索）以及结构化输出等方面进行了全面升级与统一。

---

### 环境准备

确保已安装最新版官方 Python SDK 并配置 API Key：

```bash
pip install --upgrade openai
export OPENAI_API_KEY="your-api-key-here"

```

---

### 1. 基础调用与指令分离（Instructions & Input）

在 Responses API 中，推荐将系统角色规则与用户输入拆分为 `instructions` 和 `input`。获取最终回答可直接读取 `response.output_text`。

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4o",
    instructions="你是一个资深的 Python 技术导师，请用简洁通俗的语言回答。",
    input="解释什么是生成器（Generator）？"
)

# 直接通过 output_text 读取纯文本回答
print(response.output_text)

```

---

### 2. 流式传输（Streaming）

Responses API 的流式响应以事件流形式输出，可以通过监听 `text.delta` 实时获取增量内容：

```python
from openai import OpenAI

client = OpenAI()

stream = client.responses.create(
    model="gpt-4o",
    input="写一段关于未来科技的短诗。",
    stream=True
)

for event in stream:
    if hasattr(event, "type") and "text.delta" in event.type:
        print(event.delta, end="", flush=True)
print()

```

---

### 3. 多模态输入（图片分析）

多模态输入格式更为简洁，使用 `input_image` 传递图片链接或 Base64 编码：

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4o",
    input=[
        {"role": "user", "content": "这张图片里有哪些关键元素？"},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
            ]
        }
    ]
)

print(response.output_text)

```

---

### 4. 结构化输出（Structured Outputs）

配合 Pydantic，可以确保模型强制以结构化数据输出，无需手动解析 JSON 格式：

```python
from openai import OpenAI
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    skills: list[str]

client = OpenAI()

response = client.responses.create(
    model="gpt-4o",
    input="从文本中提取信息：李雷今年28岁，精通 Python、Rust 和 Docker。",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "UserProfile",
            "schema": UserProfile.model_json_schema(),
            "strict": True
        }
    }
)

print(response.output_text)

```

---

### 5. 内置平台工具（以网络搜索 Web Search 为例）

Responses API 内置了官方托管的工具（如 `web_search_preview` / `web_search`、`file_search`），模型可自主发起搜索并自动附加引文信息（Annotations）：

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4o",
    input="2026年最近有什么重大航天发射任务吗？",
    tools=[{"type": "web_search_preview"}]
)

print(response.output_text)

```

---

### 6. 自定义函数调用（Function Calling）

模型可根据用户意图输出调用外部函数的参数，执行后再回传给模型输出最终结论：

```python
import json
from openai import OpenAI

client = OpenAI()

# 1. 模拟本地业务函数
def get_current_weather(location: str):
    return json.dumps({"location": location, "temperature": "22°C", "condition": "晴"})

# 2. 定义工具 Schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名称，如：北京"}
                },
                "required": ["location"]
            }
        }
    }
]

# 3. 初始请求
response = client.responses.create(
    model="gpt-4o",
    input="今天杭州天气怎么样？",
    tools=tools
)

# 4. 判断是否触发了工具调用并回传结果
for item in response.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
        call_result = get_current_weather(**args)

        # 5. 将函数执行结果返回给 API 获取最终总结回答
        followup = client.responses.create(
            model="gpt-4o",
            previous_response_id=response.id,  # 关联上下文会话 ID
            input=[
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": call_result
                }
            ]
        )
        print(followup.output_text)

```

---

### 关键对比速查

| 特性 | 传统 Chat Completions | 新版 Responses API |
| --- | --- | --- |
| **主入口** | `client.chat.completions.create` | `client.responses.create` |
| **文本提取** | `response.choices[0].message.content` | `response.output_text` |
| **系统提示词** | `messages: [{"role": "system", ...}]` | 专用顶级参数 `instructions="..."` |
| **托管工具** | 需要自己写外部搜索/Rag 并多次往返 | 原生内置 `web_search`、`file_search` |
| **多轮关联** | 必须在客户端完整维护并反复上传 `messages` 数组 | 支持通过 `previous_response_id` 自动延续状态 |