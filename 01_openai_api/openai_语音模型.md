OpenAI 提供了语音合成接口（Text-to-Speech，简称 TTS），可以通过官方 SDK 将文本快速转化为逼真的人声音频。以下是使用 OpenAI Audio TTS API 的完整使用教程与核心功能介绍。

---

### 一、准备工作

1. 安装官方 Python SDK：
```bash
pip install --upgrade openai

```


2. 配置环境变量 `OPENAI_API_KEY`（或在代码中显式传入）：
```bash
export OPENAI_API_KEY="your-api-key-here"

```



---

### 二、基础调用示例

调用 `client.audio.speech.create` 接口生成音频，并使用 `stream_to_file` 或 `write_to_file` 保存为本地文件：

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI()

speech_file_path = Path("output.mp3")

# 创建音频
response = client.audio.speech.create(
    model="tts-1",           # 可选: tts-1 (低延迟) 或 tts-1-hd (高保真)
    voice="alloy",           # 发音音色
    input="你好！欢迎使用 OpenAI 的语音合成服务，这是一段测试音频。"
)

# 保存到本地
response.stream_to_file(speech_file_path)
print(f"音频已成功保存至: {speech_file_path}")

```

---

### 三、核心参数与选项说明

| 参数名 | 类型 / 范围 | 默认值 | 详细说明 |
| --- | --- | --- | --- |
| `model` | `string` | 必填 | 可选 `tts-1`（针对实时低延迟优化）、`tts-1-hd`（音质更高、细节更丰富） 或最新的 `gpt-4o-mini-tts`。 |
| `input` | `string` | 必填 | 需要朗读的文本，单次请求上限通常为 **4,096 字符**。 |
| `voice` | `string` | 必填 | 内置音色，包括 `alloy`、`echo`、`fable`、`onyx`、`nova`、`shimmer`、`ash`、`coral`、`sage` 等。 |
| `response_format` | `string` | `"mp3"` | 输出格式，支持 `mp3`、`opus`、`aac`、`flac`、`wav`、`pcm`。 |
| `speed` | `float` (0.25 ~ 4.0) | `1.0` | 控制语速，1.0 为正常速度，值越小语速越慢，越大越快。 |
| `instructions` | `string` | 可选 | *（仅部分新版模型如 `gpt-4o-mini-tts` 支持）* 提示语气、情绪和口音。 |

---

### 四、进阶功能与场景

#### 1. 调整语速与音频格式

如果需要用于网络实时通话或低延迟传输，通常推荐 `opus` 格式；如果需要无损用于后期制作，可以选择 `wav` 或 `flac`。

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI()

response = client.audio.speech.create(
    model="tts-1-hd",
    voice="nova",
    input="请注意，这是一段加速播报的内容，格式为 WAV 无损音频。",
    speed=1.25,               # 1.25倍速
    response_format="wav"     # 保存为 wav 格式
)

response.stream_to_file(Path("speed_test.wav"))

```

#### 2. 流式获取并直接消费音频流（Chunk Streaming）

如果想边生成边播放（或作为服务端转发给前端），可以使用 `iter_bytes()` 进行逐块读取：

```python
from openai import OpenAI

client = OpenAI()

response = client.audio.speech.create(
    model="tts-1",
    voice="echo",
    input="流式响应能够极大减少客户端的首包等待时间。",
    response_format="mp3"
)

# 逐块获取二进制音频流
with open("streamed_output.mp3", "wb") as f:
    for chunk in response.iter_bytes(chunk_size=1024):
        f.write(chunk)

```

#### 3. 支持情绪控制的新模型指令（如果使用 `gpt-4o-mini-tts`）

在使用支持指令的 TTS 模型时，可以通过 `instructions` 参数控制播音员的情绪、停顿及语气风格：

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI()

response = client.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="alloy",
    input="嘘，小声点，大家都在休息呢。",
    instructions="用温柔低沉的耳语音调说话，带有自然的停顿"
)

response.stream_to_file(Path("whisper.mp3"))

```

---

### 五、使用技巧与最佳实践

1. **多语言支持**：TTS 模型会自动根据输入的文本语言识别并读出对应发音（支持中文、英文、日文、德文等几十种语言），无需额外指定语言参数。
2. **长文本分段（Chunking）**：单次请求上限为 4,096 字符。若需要生成整本书籍或长文章，建议在句子标点符号（如句号、感叹号、换行符）处切分为多个片段依次调用，并在客户端通过音频处理库（如 `pydub`）进行拼接。
3. **音色试听对比**：`alloy` 和 `nova` 声音亲和力强，适合导读与助手；`onyx` 和 `echo` 声线深沉有力，适合纪录片或新闻播报；建议在正式集成前用同一段业务文本先测试对比不同 Voice 的听觉效果。