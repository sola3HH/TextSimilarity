import re
import os
import nltk
from gensim import corpora


def tokenization(fileName):
    """对目标文档进行分词处理"""
    with open(fileName, 'r', encoding='utf-8') as file:
        u = file.read()
        str = re.sub('[^\w ]', '', u)
        return nltk.word_tokenize(str)


def cut_word(str):
    return nltk.word_tokenize(str)


# 确认文件路径
filePath = 'texts/'
# 分词
texts = []
for fileName in os.listdir(filePath):
    texts.append(tokenization('texts/' + fileName))
# 建立词袋并向量化（语料库）
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2idx(text) for text in texts]

# 构建原文本
origin = tokenization('texts/' + '标准答案.txt')
origin_bow = dictionary.doc2bow(origin)
origin_vector = []
# 构建原文本词频向量
for i in range(0, len(dictionary)):
    flag = 0
    for index in origin_bow:
        if index[0] == i:
            origin_vector.append(index[1])
            flag = 1
            break
    if flag == 0:
        origin_vector.append(0)

# 遍历构建构建查询文本
for query_text in os.listdir(filePath):
    if query_text == '标准答案.txt':
        continue
    query = tokenization('texts/' + query_text)
    query_bow = dictionary.doc2bow(query)
    query_vector = []
    # 构建查询文本词频向量
    for i in range(0, len(dictionary)):
        flag = 0
        for index in query_bow:
            if index[0] == i:
                query_vector.append(index[1])
                flag = 1
                break
        if flag == 0:
            query_vector.append(0)

    # 余弦相似度
    A, B, AB = 0, 0, 0
    for value in origin_vector:
        A += value ** 2

    for value in query_vector:
        B += value ** 2

    for i in range(0, len(dictionary)):
        AB += origin_vector[i] * query_vector[i]

    sim = AB / ((A ** 0.5) * (B ** 0.5))

    print('文本' + query_text + '与原文的相似度为：%.2f' % (sim * 100) + "%")
