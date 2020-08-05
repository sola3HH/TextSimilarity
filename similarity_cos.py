import re

import nltk
from gensim import corpora


def tokenization(fileName):
    """对目标文档进行分词处理"""
    with open(fileName, 'r', encoding='utf-8') as file:
        u = file.read()
        str = re.sub('[^\w ]', '', u)
        return nltk.word_tokenize(str)


# 分词
texts = [tokenization('test1.txt'), tokenization('test2.txt')]
# 建立词袋并向量化（语料库）
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2idx(text) for text in texts]
# 构建原文本
origin = tokenization('test1.txt')
origin_vector = dictionary.doc2idx(origin)
# 构建query文本
query = tokenization('test2.txt')
query_vector = dictionary.doc2idx(query)

# 余弦相似度
A, B = 0, 0
for value in origin_vector:
    A += value ** 2

for value in query_vector:
    B += value ** 2

sim = A / ((A ** 0.5) * (B ** 0.5))

print('两篇文章的相似度为：%.2f' % (sim * 100) + "%")
