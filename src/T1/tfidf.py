import jieba
from gensim.similarities import SparseMatrixSimilarity
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import LdaModel
from src.T1.evaluation import F1Score

import numpy as np
import xlrd


def similar_result(sentence, key):
    # 1、基于文本集建立词典，并获得词典特征数
    all_dict = sentence
    content_dict = Dictionary(all_dict)
    # 2、基于文本集建立词典，并获得词典特征数
    num_features = len(content_dict.token2id)
    # 3.1、基于词典，将分词列表集转换成稀疏向量集，称作语料库
    corpus = [content_dict.doc2bow(sen) for sen in sentence]
    # 3.2、同理，用词典把搜索词也转换为稀疏向量
    kw_vector = content_dict.doc2bow(key)
    # 4、创建TF-IDF模型，传入语料库来 训练
    tfidf = TfidfModel(corpus)
    # tfidf = LdaModel(corpus=corpus, id2word=content_dict, num_topics=1000)  # 使用lda模型
    # 5、用训练好的TF-IDF模型处理被检索文本和搜索词
    tf_texts = tfidf[corpus]
    # 此处将语料库用作被检索文本
    tf_kw = tfidf[kw_vector]
    sparse_matrix = SparseMatrixSimilarity(tf_texts, num_features)
    similaritie = sparse_matrix.get_similarities(tf_kw)
    result = []
    for e, s in enumerate(similaritie, 1):
        result.append(s)
    return result


def solve():
    message = np.load('message.npy', allow_pickle=bool).item()
    classifications = np.load('classifications.npy', allow_pickle=bool).item()

    keys = list(message.values())
    ans = list(classifications)
    # 将字典的value转换成列表
    sent = list(classifications.values())

    list_mine = []
    for k in keys:
        grade = similar_result(sent, k)
        # print(grade, np.argsort(grade))
        list_mine.append(ans[np.argsort(grade)[-1]])

    path = 'data/messages.xlsx'
    ExcelFile1 = xlrd.open_workbook(path)
    sheet1 = ExcelFile1.sheet_by_index(0)
    list_ans = []
    for i in range(1, sheet1.nrows):
        text = sheet1.cell(i, 5).value.encode('utf-8').decode('utf-8-sig')
        list_ans.append(text)

    for i in range(len(list_ans)):
        if list_ans[i] != list_mine[i]:
            print(i, list_mine[i], list_ans[i])

    return F1Score().evaluate(list_ans, list_mine)


def main():
    result = solve()
    print(result)


if __name__ == "__main__":
    main()
