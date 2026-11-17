下面把你需要掌握的两个方法整理成一份 API 速查笔记：

1. `tokenizer.encode()`：将文本编码为 Token ID。
2. `tokenizer()`：完整的文本编码接口，支持批量处理、填充、截断以及返回模型输入信息。

参数以 Hugging Face 当前官方文档为参考。不同 Transformers 版本可能存在细微差异。

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face+1



# 一、encode() 方法

## 1. 方法签名

```
tokenizer.encode(    text,    text_pair=None,    add_special_tokens=True,    padding=False,    truncation=None,    max_length=None,    stride=0,    padding_side=None,    return_tensors=None,    **kwargs)
```

## 2. 所有参数

| 参数                   | 类型                | 默认值   | 作用                 |
| -------------------- | ----------------- | ----- | ------------------ |
| `text`               | str / list        | 必填    | 第一段待编码文本           |
| `text_pair`          | str / list / None | None  | 第二段文本，用于句子对任务      |
| `add_special_tokens` | bool              | True  | 是否添加特殊 Token       |
| `padding`            | bool / str        | False | 是否填充序列             |
| `truncation`         | bool / str / None | None  | 是否截断超长序列           |
| `max_length`         | int / None        | None  | 最大序列长度             |
| `stride`             | int               | 0     | 溢出文本块之间的重叠 Token 数 |
| `padding_side`       | str / None        | None  | 填充方向，left 或 right  |
| `return_tensors`     | str / None        | None  | 返回 Tensor 的类型      |
| `**kwargs`           | dict              | —     | 其他底层 Tokenizer 参数  |

## 3. 重点参数详解

① text 和 text_pair

```
tokenizer.encode(    "今天天气很好",    text_pair="我想出去玩")
```

用于单句或句子对编码。

② add_special_tokens

```
add_special_tokens=True
```

是否添加 `[CLS]`、`[SEP]` 等模型规定的特殊 Token。

③ padding

| 值              | 含义             |
| -------------- | -------------- |
| `False`        | 不填充            |
| `True`         | 填充到批次最长长度      |
| `"longest"`    | 同 True         |
| `"max_length"` | 填充到 max_length |
| `"do_not_pad"` | 不填充            |

④ truncation

| 值                 | 含义                  |
| ----------------- | ------------------- |
| `False`           | 不截断                 |
| `True`            | 默认 longest_first 策略 |
| `"longest_first"` | 优先截断较长序列            |
| `"only_first"`    | 只截断第一段              |
| `"only_second"`   | 只截断第二段              |

⑤ max_length

控制最大 Token 数量，通常与 `truncation` 或 `padding` 配合。

⑥ stride

控制长文本切分时，相邻文本块之间的重叠数量。

⑦ padding_side

- `left`：左侧填充。
- `right`：右侧填充。

⑧ return_tensors

| 值       | 返回类型              |
| ------- | ----------------- |
| `None`  | Python 列表         |
| `"pt"`  | PyTorch Tensor    |
| `"tf"`  | TensorFlow Tensor |
| `"np"`  | NumPy 数组          |
| `"jax"` | JAX 数组（视版本支持情况）   |

注意：`encode()` 的主要返回结果是 Token ID，不会像 `tokenizer()` 一样完整返回 `attention_mask`、`token_type_ids` 等字段。

# 二、tokenizer() 方法

## 1. 方法签名

当前版本的核心签名如下：

```
tokenizer(    text=None,    text_pair=None,    text_target=None,    text_pair_target=None,    add_special_tokens=True,    padding=False,    truncation=None,    max_length=None,    stride=0,    is_split_into_words=False,    pad_to_multiple_of=None,    padding_side=None,    return_tensors=None,    return_token_type_ids=None,    return_attention_mask=None,    return_overflowing_tokens=False,    return_special_tokens_mask=False,    return_offsets_mapping=False,    return_length=False,    verbose=True,    tokenizer_kwargs=None,    **kwargs)
```

## 2. 所有参数

| 参数                           | 类型                | 默认值   | 作用                 |
| ---------------------------- | ----------------- | ----- | ------------------ |
| `text`                       | str / list / None | None  | 输入文本或文本批次          |
| `text_pair`                  | str / list / None | None  | 第二段文本              |
| `text_target`                | str / list / None | None  | 目标文本，常用于生成任务       |
| `text_pair_target`           | str / list / None | None  | 目标文本的第二段           |
| `add_special_tokens`         | bool              | True  | 是否添加特殊 Token       |
| `padding`                    | bool / str        | False | 填充序列               |
| `truncation`                 | bool / str / None | None  | 截断序列               |
| `max_length`                 | int / None        | None  | 最大长度               |
| `stride`                     | int               | 0     | 重叠 Token 数         |
| `is_split_into_words`        | bool              | False | 输入是否已经预先分词         |
| `pad_to_multiple_of`         | int / None        | None  | 填充到指定整数倍           |
| `padding_side`               | str / None        | None  | 填充方向               |
| `return_tensors`             | str / None        | None  | 返回 Tensor 类型       |
| `return_token_type_ids`      | bool / None       | None  | 是否返回句子类型 ID        |
| `return_attention_mask`      | bool / None       | None  | 是否返回注意力掩码          |
| `return_overflowing_tokens`  | bool              | False | 是否返回溢出的 Token      |
| `return_special_tokens_mask` | bool              | False | 是否返回特殊 Token 掩码    |
| `return_offsets_mapping`     | bool              | False | 是否返回原文字符位置         |
| `return_length`              | bool              | False | 是否返回编码长度           |
| `verbose`                    | bool              | True  | 是否显示警告和提示          |
| `tokenizer_kwargs`           | dict / None       | None  | 额外传给 Tokenizer 的参数 |
| `**kwargs`                   | dict              | —     | 其他额外参数             |

以上签名和参数定义以当前官方 API 为准。

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face+1



## 3. 重点参数详解

这里有几个参数是 `tokenizer()` 比 `encode()` 更值得学习的地方。

① text：输入文本

支持单条文本：

```
tokenizer("我正在学习 NLP")
```

也支持批量文本：

```
tokenizer([    "我正在学习 NLP",    "我正在学习 PyTorch"])
```

还支持预先分词的文本列表，具体见 `is_split_into_words`。

② text_pair：第二段文本

```
tokenizer(    "今天天气很好",    text_pair="我想出去玩")
```

主要用于 BERT 句子对任务，例如文本匹配、自然语言推理。

③ text_target：目标文本

主要用于序列到序列（Seq2Seq）模型。

例如机器翻译：

```
result = tokenizer(    text="I love machine learning",    text_target="我喜欢机器学习")
```

这里：

- `text` 是模型的输入文本。
- `text_target` 是模型需要学习生成的目标文本。

④ text_pair_target：目标文本的第二段

用于需要编码目标文本句子对的特殊场景，普通 NLP 入门暂时不需要掌握。

⑤ add_special_tokens

是否添加特殊 Token。

例如 BERT：

```
[CLS] 我 喜欢 NLP [SEP]
```

⑥ padding

控制序列填充方式。

```
padding=True
```

填充到当前批次最长文本。

```
padding="max_length"
```

填充到指定的 `max_length`。

⑦ truncation

控制超长文本截断方式：

- `True`：启用默认截断策略。
- `"longest_first"`：优先截断较长的序列。
- `"only_first"`：只截断第一段。
- `"only_second"`：只截断第二段。

⑧ max_length

控制最大 Token 数量。

```
tokenizer(    "我正在学习自然语言处理",    truncation=True,    max_length=128)
```

⑨ stride

控制长文本切分时的重叠 Token 数量。

例如：

```
tokenizer(    long_text,    max_length=100,    truncation=True,    stride=20,    return_overflowing_tokens=True)
```

每个后续文本块会与前一个文本块重叠 20 个 Token。

这在 RAG 文档切片中非常有用。

⑩ is_split_into_words

告诉 Tokenizer：输入是否已经经过预分词。

```
words = ["我", "喜欢", "学习", "机器学习"]result = tokenizer(    words,    is_split_into_words=True)
```

特别适用于：

- NER（命名实体识别）
- 词性标注
- Token Classification（Token 分类）

⑪ pad_to_multiple_of

将填充后的长度调整为指定数字的倍数。

例如：

```
tokenizer(    texts,    padding=True,    pad_to_multiple_of=8)
```

如果当前批次最长长度为 13，可能会填充到 16。

在 GPU 训练中，有时可以帮助提高计算效率。

⑫ padding_side

控制填充方向：

```
tokenizer.padding_side = "right"
```

或者：

```
tokenizer.padding_side = "left"
```

⑬ return_tensors

决定返回的数据类型：

```
return_tensors="pt"
```

返回 PyTorch Tensor。

⑭ return_token_type_ids

是否返回句子类型标识。

```
return_token_type_ids=True
```

例如 BERT 句子对：

```
[CLS] 句子A [SEP] 句子B [SEP]
  0      0     0      1       1
```

⑮ return_attention_mask

是否返回注意力掩码。

```
return_attention_mask=True
```

例如：

```
input_ids:
[101, 2769, 5543, 102, 0, 0]

attention_mask:
[  1,    1,    1,   1, 0, 0]
```

1 表示有效 Token，0 表示 Padding。

⑯ return_overflowing_tokens

是否返回截断后剩余的 Token。

```
return_overflowing_tokens=True
```

常与 `stride`、`max_length` 配合，用于长文本分块。

注意：对于句子对使用 `longest_first` 截断时，该参数存在限制，可能报错。

⑰ return_special_tokens_mask

返回特殊 Token 的掩码。

```
return_special_tokens_mask=True
```

例如：

```
input_ids:
[101, 2769, 5543, 102]

special_tokens_mask:
[  1,    0,    0,   1]
```

1 表示特殊 Token，0 表示普通 Token。

⑱ return_offsets_mapping

返回 Token 对应的原始文本字符位置。

```
return_offsets_mapping=True
```

例如：

```
text = "我喜欢机器学习"
```

可以获得每个 Token 在原始字符串中的起止位置。

常用于：

- NER 实体位置映射
- 文本高亮
- Token 与原文对齐

注意：通常只有 Fast Tokenizer 支持此功能。

⑲ return_length

是否返回编码后的长度。

```
return_length=True
```

结果中会增加 `length` 字段。

⑳ verbose

是否输出相关警告和提示信息。

```
verbose=False
```

通常保持默认即可。

㉑ tokenizer_kwargs

用于传入额外的 Tokenizer 配置字典。显式参数优先级更高。

㉒ kwargs

额外参数，具体取决于底层 Tokenizer 实现。

# 三、两个方法的核心区别

| 对比项目           | encode()      | tokenizer() |
| -------------- | ------------- | ----------- |
| 单条文本           | 支持            | 支持          |
| 批量文本           | 不作为主要用途       | 支持          |
| 句子对            | 支持            | 支持          |
| 特殊 Token       | 支持            | 支持          |
| Padding        | 支持相关参数        | 支持          |
| Truncation     | 支持相关参数        | 支持          |
| Attention Mask | 不直接返回         | 支持          |
| Token Type IDs | 不直接返回         | 支持          |
| Offset Mapping | 不直接返回         | 支持          |
| 长文本溢出分块        | 不适合作为主要接口     | 支持          |
| 推荐用途           | 单纯获取 Token ID | 实际模型输入      |

# 四、实际开发中怎么选择？

我给你总结三个最常见的场景。

场景一：只想知道文本转换成什么数字

```
ids = tokenizer.encode("我喜欢学习 NLP")
```

使用 encode()

场景二：准备数据输入 BERT

```
inputs = tokenizer(    texts,    padding=True,    truncation=True,    max_length=128,    return_tensors="pt")
```

使用 tokenizer()

场景三：RAG 长文档切片

```
chunks = tokenizer(    long_text,    max_length=256,    truncation=True,    stride=50,    return_overflowing_tokens=True)
```

使用 tokenizer()

# 五、给你的学习重点

我建议暂时不要死记所有参数，而是分成三个阶段：

| 学习阶段 | 掌握内容                                                                |
| ---- | ------------------------------------------------------------------- |
| 第一阶段 | `text`、`add_special_tokens`、`encode()`、`decode()`                   |
| 第二阶段 | `padding`、`truncation`、`max_length`、`return_tensors`                |
| 第三阶段 | `attention_mask`、`token_type_ids`、`stride`、`return_offsets_mapping` |

尤其要理解 `input_ids`、`attention_mask` 和 `token_type_ids` 之间的关系。

官方参考文档：

- [Tokenizer API 官方文档](https://huggingface.co/docs/transformers/main_classes/tokenizer?highlight=pretrainedtokenizer\&utm_source=chatgpt.com)
- [encode() API 官方文档](https://huggingface.co/docs/transformers/main/internal/tokenization_utils?utm_source=chatgpt.com)

下一步建议： 我们可以拿 `bert-base-chinese` 做一次完整实验，把同一段中文分别经过 `encode()` 和 `tokenizer()`，逐个比较它们的输出，真正理解这些参数是如何影响模型输入的。