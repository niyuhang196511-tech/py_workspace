from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
base_url = os.getenv("BL_BASE_URL")
api_key = os.getenv("BL_API_KEY")
model_name = 'qwen3.8-flash'

client = OpenAI(base_url=base_url, api_key=api_key)

previous_response_id = None  # 改名以示区分

while True:
    _i = input("请输入(按Q/q退出): ")

    if not _i.strip():  # 空输入或纯空格，跳过
        print("输入不能为空，请重新输入～")
        continue

    if _i.strip().lower() == 'q':
        break

    stream = client.responses.create(
        model=model_name,
        temperature=0.5,
        stream=True,
        instructions="你是一个语言大师,无论问你什么问题,你有可以高情商的进行解答",  # 改为 instructions
        input=_i,
        previous_response_id=previous_response_id  # 参数名修正
    )

    current_response_id = None

    for event in stream:
        if event.type == "response.created":
            # 在这里拿到本轮响应的 ID，供下一轮使用
            current_response_id = event.response.id

        elif event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)

        elif event.type == "response.completed":
            print("\n=================================================")
            # 也可以在这里通过 event.response.id 获取，更稳妥
            # current_response_id = event.response.id

        elif event.type == "error":
            print(f"\n[错误]: {event.message}")

    # 一轮结束后，更新 previous_response_id
    if current_response_id:
        previous_response_id = current_response_id