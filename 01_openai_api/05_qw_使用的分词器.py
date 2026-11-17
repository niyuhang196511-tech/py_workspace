from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-Flash-Next", trust_remote_code=True)

res1 = tokenizer.encode("China is great!")
print(res1)

res2 = tokenizer.encode("你好, 世界!")
print(res2)

res3 = tokenizer.decode(res1)
print(res3)

res4 = tokenizer.decode(res2)
print(res4)


words = [tokenizer.decode(x) for x in res2]
print(words)