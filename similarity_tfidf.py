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
texts = [tokenization('test1.txt'), tokenization('test2.txt'), tokenization('test3.txt')]
# 建立词袋并向量化（语料库）
dictionary = corpora.Dictionary(texts)
num_features = len(dictionary.token2id)
corpus = [dictionary.doc2bow(text) for text in texts]
# 训练TF-IDF模型
tfidf = models.TfidfModel(corpus)
# 构建query文本
query = tokenization('标准答案.txt')
query_vector = dictionary.doc2bow(query)
# 对稀疏向量建立索引
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features)
# 相似计算
sim = index[tfidf[query_vector]]

result_list = []
for i in range(len(sim)):
    print('与第%d个文件相似度为: %.2f' % (i + 1, sim[i] * 100) + '%')
