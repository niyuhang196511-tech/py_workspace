import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

from narwhals import DataFrame
from openai import OpenAI

plt.rcParams['font.sans-serif'] = ['sans-serif'] # 用来正常吸纳是中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示中文字符

df = pd.read_csv("./datas/embedding_output_1k.csv")

df['embedding_vec'] = df['embedding'].apply(ast.literal_eval)

embedding_model = "qwen3.7-text-embedding"

client = OpenAI(
    base_url=os.getenv("BL_BASE_URL"),
    api_key=os.getenv("BL_API_KEY"),
)

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

    return resp.data[0].embedding


def consine_distance(vec1, vec2):
    """
    计算两个向量之间的余弦距离
    :param vec1:
    :param vec2:
    :return:
    """
    return np.dot(vec1, vec1) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def search_by_word(df: DataFrame, word_key, top = 3, print_flag = True):
    """
    根据指定的关键词去向量空间中进行相似搜索
    :param df:
    :param word_key:
    :param top:
    :param print_flag:
    :return:
    """
    # 向量化查询关键词
    word_embedding = embedding_text(word_key)

    # 计算相似度
    df['similarity'] = df.embedding_vec.apply(lambda x: consine_distance(x, word_embedding))

    result = df.sort_values('similarity', ascending=False).head(top)

    if print_flag:
        for r in result.itertuples():
            print("Summary: {}; Text: {}".format(r.Summary, r.Text))
if __name__ == '__main__':
    search_by_word(df, 'delicious beans')

