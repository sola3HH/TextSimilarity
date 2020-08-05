import re

import nltk
from gensim import corpora, models, similarities


def tokenization(fileName):
    """对目标文档进行分词处理"""
    with open(fileName, 'r', encoding='utf-8') as file:
        u = file.read()
        str = re.sub('[^\w ]', '', u)
        return nltk.word_tokenize(str)


# 分词
texts = [tokenization('test1.txt')]
# 建立词袋并向量化（语料库）
dictionary = corpora.Dictionary(texts)
feature_cnt = len(dictionary.token2id.keys())
corpus = [dictionary.doc2bow(text) for text in texts]
# 建立TF-IDF模型
tfidf = models.TfidfModel(corpus)
# 构建query文本
query = tokenization('test2.txt')
query_vector = dictionary.doc2bow(query)
# 对稀疏向量建立索引
index = similarities.MatrixSimilarity(tfidf[corpus], num_features=feature_cnt)
# 相似计算
sim = index[tfidf[query_vector]]
print(sim)
result_list = []
for i in range(len(sim)):
    print('与原文件相似度为: %.2f' % (sim[i]))