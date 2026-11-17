OpenAI **Responses API** 本身是统一的大语言模型多模态交互接口，而图片生成属于图像输出能力。

在 OpenAI 架构中，生成图片有两种核心实现方式：

1. **直接调用独立的 Images API (`client.images.generate`)**：最标准的图片生成模式。
2. **在 Responses API 中通过 Function Calling（工具调用）触发图片生成**：让模型在对话或任务中根据上下文“自主决定”何时画图。

以下是完整的两种实操教程：

---

### 方法一：标准原生方式（Images API）

直接生成图片时，推荐使用 `client.images.generate` 与 `dall-e-3` 模型：

#### 1. 基础文字生成图片

```python
from openai import OpenAI

client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="一个坐在木质书桌前看书的宇航员，窗外是深邃的银河星空，写实风格，8k分辨率",
    size="1024x1024",     # 分辨率：1024x1024, 1024x1792 (竖屏), 1792x1024 (横屏)
    quality="standard",    # 画质：standard 或 hd
    n=1                   # dall-e-3 单次请求限制为 1
)

# 打印生成的临时访问链接（链接有时效性，约 1-2 小时）
print("图片链接:", response.data[0].url)

```

#### 2. 直接下载并保存到本地（Base64 模式）

通过将 `response_format` 设为 `b64_json`，无需通过网络下载 URL，可直接存为本地文件：

```python
import base64
from openai import OpenAI

client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="赛博朋克风格的雨夜小巷，霓虹灯倒影在水洼中",
    response_format="b64_json"
)

# 解码 base64 数据并保存
image_bytes = base64.b64decode(response.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)

print("图片已保存至本地 output.png")

```

---

### 方法二：在 Responses API 中结合 Tool Calling 生成图片

如果希望在 **Responses API** 的统一对话交互流程中实现“模型根据提示词自动调用画图功能”，可以将生成图片封装成一个工具（Tool）：

```python
import json
import base64
from openai import OpenAI

client = OpenAI()

# 1. 定义实际的画图本地函数
def generate_image_tool(prompt: str) -> str:
    img_resp = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024"
    )
    return img_resp.data[0].url

# 2. 声明工具 Schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "当用户需要生成、绘制图像或视觉设计时调用此函数",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "详细的画图提示词（尽量丰富构图、光影和风格细节）"
                    }
                },
                "required": ["prompt"]
            }
        }
    }
]

# 3. 使用 Responses API 提问
response = client.responses.create(
    model="gpt-4o",
    instructions="你是一个设计助手，如果用户需要绘图，请调用 generate_image 工具。",
    input="帮我画一张在火星表面种植植物的插画",
    tools=tools
)

# 4. 检测并执行画图工具调用
for item in response.output:
    if item.type == "function_call" and item.name == "generate_image":
        args = json.loads(item.arguments)
        print(f"模型生成的画图提示词: {args['prompt']}")
        
        # 执行画图
        image_url = generate_image_tool(args["prompt"])
        print(f"生成的图片链接: {image_url}")

        # 5. 将图片链接结果回传给 Responses API，完成对话闭环
        followup = client.responses.create(
            model="gpt-4o",
            previous_response_id=response.id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps({"image_url": image_url})
                }
            ]
        )
        print("助手总结:", followup.output_text)

```

---

### 核心参数与限制说明

* **尺寸支持**：`dall-e-3` 支持 `1024x1024`（正方形）、`1792x1024`（横版比例）和 `1024x1792`（竖版比例）。
* **生成张数**：`dall-e-3` 的 `n` 参数只允许填 `1`；如需单次批量生成多张图片，需使用 `dall-e-2`。
* **时效性提示**：返回的 `url` 有过期时间，生产环境中建议选择 `b64_json` 格式并在服务器端直接上传至你的对象存储（如 OSS / S3）。