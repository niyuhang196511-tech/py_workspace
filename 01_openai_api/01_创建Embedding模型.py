import pandas as pd
import os
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv

load_dotenv(verbose=True)

df = pd.read_csv("./datas/fine_food_reviews.csv", index_col=0)

df = df[
    ['ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time', 'Summary',
     'Text']]

df = df.dropna()

df['combined'] = "Title: " + df.Summary.str.strip() + ";Content: " + df.Text.str.strip()

print(df.head(2))

# 生成Embedding之后的向量空间，并且保存
embedding_model = "qwen3.7-text-embedding"

# 对文本进行处理的分词器
tokenizer_name = "cl100k_base"
max_tokens = 128000
top_n = 1000
df = df.sort_values('Time')
df.drop('Time', axis=1, inplace=True)

# 创建一个分词器
tokenizer = tiktoken.get_encoding(encoding_name=tokenizer_name)

df['count_token'] = df.combined.apply(lambda x: len(tokenizer.encode(x)))

df = df[df['count_token'] <= max_tokens].tail(top_n)

print(len(df))

client = OpenAI(
    base_url=os.getenv("BL_BASE_URL"),
    api_key=os.getenv("BL_API_KEY"),
)

flag = True


def embedding_text(text: str, model: str = embedding_model):
    """
    通过OpenAI的Embedding模型处理文本数据
    :param text: 需要处理的文本数据
    :param model: 模型
    :return:
    """
    resp = client.embeddings.create(
        input=text,
        model=model,
    )

    global flag
    if flag:
        print(resp)
        flag = False

    return resp.data[0].embedding


df['embedding'] = df['combined'].apply(embedding_text)

df.to_csv("./datas/embedding_output_1k.csv")
