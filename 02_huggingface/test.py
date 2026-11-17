# 1. 先配置环境变量
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
os.environ["HF_HUB_VERBOSITY"] = "info"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "15"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

# 2. 再导入 Transformers
from transformers import AutoTokenizer

# 3. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-chinese"
)

# 4. 测试
print(tokenizer.tokenize("我正在学习自然语言处理"))