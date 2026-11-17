from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
base_url = os.getenv("BL_BASE_URL")
api_key = os.getenv("BL_API_KEY")
model_name = 'qwen3.8-flash'

client = OpenAI(base_url=base_url, api_key=api_key)

assistant = client.responses.create(
    name="Lao Xiao",
    model=model_name,
    instructions="你是数学家,可以为我解答任何数学问题和进行数学计算",
    tools=[{'type': 'code_interpreter'}],
)