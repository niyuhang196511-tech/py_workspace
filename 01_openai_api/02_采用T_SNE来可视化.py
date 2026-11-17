import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE

import ast

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常吸纳是中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示中文字符

df = pd.read_csv("./datas/embedding_output_1k.csv")

df['embedding_vec'] = df['embedding'].apply(ast.literal_eval)

print(df.head(2))

# T-SNE 可以将高纬度的数据映射到2D / 3D的空间中，以便我们进行观察。

if df['embedding_vec'].apply(len).nunique() == 1:
    matrix = np.vstack(df['embedding_vec'].values)

    # 创建一个 T-SNE模型
    tsne = TSNE(n_components=2,random_state=42, init='random', learning_rate=200)

    matrix_2d = tsne.fit_transform(matrix)

    print(matrix_2d)

    # 可视化
    colors = ['red', 'darkorange', 'gold', 'turquoise', 'darkgreen']

    x = matrix_2d[:, 0]
    y = matrix_2d[:, 1]

    colors_indices = df['Score'].values - 1

    color_map = matplotlib.colors.ListedColormap(colors)

    plt.scatter(x=x, y=y, c=colors_indices, cmap=color_map, alpha=0.3)

    plt.title("使用T-SNE降维后的评论")

    plt.show()