import os
import re
import sys
import urllib.request
import nltk
import xlrd, xlwt
import requests
from gensim import corpora


def tokenization(fileName):
    """对目标文档进行分词处理"""
    with open(fileName, 'r', encoding='utf-8') as file:
        u = file.read()
        fragments = re.sub('[^\w ]', '', u)
        return nltk.word_tokenize(fragments)


def cut_word(sentence):
    fragments = re.sub('[^\w ]', '', sentence)
    return nltk.word_tokenize(fragments)


# 确认文件路径
filePath = 'texts/'
originFileName = '原文.txt'
files = os.listdir(filePath)

print("选择获取对比文件的方式：")
print("1. 在'/text'目录下")
print("2. 通过调用asr获取文本")
method = input("请输入1或者2\n")
if method == "1":
    if len(files) == 0:
        print(filePath)
        print('路径下为空，请检查目录')
        sys.exit()

    # 建立词袋并向量化（语料库）
    texts = []
    for fileName in files:
        texts.append(tokenization(filePath + fileName))
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2idx(text) for text in texts]

    # 构建标准文本
    if not os.path.isfile(filePath + originFileName):
        print('原文件不存在，请检查目录')
        sys.exit()
    origin = tokenization(filePath + originFileName)
    origin_bow = dictionary.doc2bow(origin)
    print('\n原文：')
    print(open(filePath + originFileName).read())

    # 构建标准文本词频向量
    origin_vector = []
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
    for query_text in files:
        if len(files) == 1 and files[0] == originFileName:
            print('对比文件不存在，请检查目录')
            sys.exit()
        if query_text == originFileName:
            continue

        query = tokenization(filePath + query_text)
        query_bow = dictionary.doc2bow(query)
        print('\n' + query_text + ':')
        print(open(filePath + query_text).read())

        # 构建查询文本词频向量
        query_vector = []
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

# 通过调取api获得返回信息
if method == "2":
    urllib.request.urlretrieve('https://www.cnblogs.com/qikeyishu/p/10748497.html', 'mp3/1.mp3')

    # # 创建新表格
    # statistic = xlwt.Workbook(encoding='utf-8')
    # sim_sheet = statistic.add_sheet('相似度对比')
    #
    # sim_sheet.write(0, 0, '语料编号')
    # sim_sheet.write(0, 2, '语料音频链接')
    # sim_sheet.write(0, 3, '原文本')
    # sim_sheet.write(0, 4, '调取结果')
    # sim_sheet.write(0, 4, '相似度')
    #
    # # 按行读取表格数据
    # table = xlrd.open_workbook('表格地址')
    # sheet = table.sheet_by_index(0)
    # for row in range(1, sheet.nrows):
    #     row_value = sheet.row(row)
    #     value1 = row_value[1].value
    #     value2 = row_value[2].value

    # 接口调取
    # recordPath = 'mp3/1.mp3'
    # url = "https://aidu-test.pingan.com/api/asr/send/direct?appId=101&appSecret=101"
    # files = [
    #     ('file', ('1.mp3', open(recordPath, 'rb'), 'audio/mpeg'))
    # ]
    # # 添加headers会导致boundary出错，传输文件时会自动带上headers
    # # headers = {
    # #     'Content-Type': 'multipart/form-data'
    # # }
    # response = requests.request("POST", url, files=files)
    # json = response.json()
    #
    # if json['code'] != 20000 or json['msg'] != '成功':
    #     print('asr接口调用出错，错误信息：')
    #     print(response.text)
    #     sys.exit()
    #
    # result = json['result']
    # # 分词
    # if not os.path.isfile(filePath + originFileName):
    #     print('原文件不存在，请检查目录')
    #     sys.exit()
    #
    # # 建立词袋并向量化（语料库）
    # texts = [tokenization(filePath + originFileName), cut_word(result)]
    # dictionary = corpora.Dictionary(texts)
    # corpus = [dictionary.doc2idx(text) for text in texts]
    #
    # # 构建标准文本
    # origin = tokenization(filePath + originFileName)
    # origin_bow = dictionary.doc2bow(origin)
    #
    # # 构建标准文本词频向量
    # origin_vector = []
    # for i in range(0, len(dictionary)):
    #     flag = 0
    #     for index in origin_bow:
    #         if index[0] == i:
    #             origin_vector.append(index[1])
    #             flag = 1
    #             break
    #     if flag == 0:
    #         origin_vector.append(0)
    #
    # # 构建查询文本
    # query = cut_word(result)
    # query_bow = dictionary.doc2bow(query)
    #
    # # 构建查询文本词频向量
    # query_vector = []
    # for i in range(0, len(dictionary)):
    #     flag = 0
    #     for index in query_bow:
    #         if index[0] == i:
    #             query_vector.append(index[1])
    #             flag = 1
    #             break
    #     if flag == 0:
    #         query_vector.append(0)
    #
    # # 余弦相似度
    # A, B, AB = 0, 0, 0
    # for value in origin_vector:
    #     A += value ** 2
    #
    # for value in query_vector:
    #     B += value ** 2
    #
    # for i in range(0, len(dictionary)):
    #     AB += origin_vector[i] * query_vector[i]
    #
    # sim = AB / ((A ** 0.5) * (B ** 0.5))
    #
    # print('\n原文：')
    # print(open(filePath + originFileName).read())
    # print('\n音频调取结果:')
    # print(result)
    #
    # print('\n音频调取结果与原文的相似度为：%.2f' % (sim * 100) + "%")
