import re
import warnings

import jieba
import jieba.analyse
import numpy as np
from gensim import corpora, models, similarities

warnings.filterwarnings('ignore')


class WordStriper(object):
    def __init__(self, stop_word_file="data/extract_dict/stop_words_dd.txt"):
        self._stop_word_file = stop_word_file
        with open(self._stop_word_file, "r") as f:
            self._stop_words = f.read().split("\n")

    def strip(self, text, HMM=True, cut_all=False):
        _text = text.replace("\n", "").replace("\t", "").replace(" ", "")

        # first replace
        for stop_word in self._stop_words:
            _text = _text.replace(stop_word, ",")
        seg_list = jieba.cut(_text, HMM=HMM, cut_all=cut_all)
        word_list = []

        # second replace
        for seg in seg_list:
            if len(seg.strip()) > 0 and seg.strip() not in self._stop_words:
                word_list.append(seg.strip())
        return list(word_list)


striper = WordStriper()


def data_analyse(key_text, total_texts):
    jieba.load_userdict("data/extract_dict/address_words.txt")
    jieba.load_userdict("data/extract_dict/platform_words.txt")
    jieba.load_userdict("data/extract_dict/sogou_dict.txt")
    jieba.load_userdict("data/extract_dict/housing_estate.txt")
    jieba.analyse.set_stop_words("data/extract_dict/stop_words_dd.txt")

    result = []
    for line in total_texts:
        result.append(striper.strip(line))

    return result, striper.strip(key_text)


def calculate_similarity(key_text, total_texts):
    # 1、将【文本集】生产【分词列表】
    texts, orig_texts = data_analyse(key_text, total_texts)

    # 一、建立词袋模型
    # 2、基于文件集建立【词典】，并提取词典特征数
    dictionary = corpora.Dictionary(texts)
    feature_cnt = len(dictionary.token2id.keys())

    # 3、基于词典，将【分词列表集】转换为【稀疏向量集】，也就是【语料库】
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 二、建立TF-IDF模型
    tfidf = models.TfidfModel(corpus)

    # 三构建一个query文本，利用词袋模型的字典将其映射到向量空间
    # 5、同理，用词典把搜索词也转换为稀疏向量
    kw_vector = dictionary.doc2bow(orig_texts)
    # 6、对稀疏向量建立索引
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)
    # 7、相似的计算
    sim = index[tfidf[kw_vector]]

    return sim


def get_word_vector(s1, s2):
    """
    :param s1: 句子1
    :param s2: 句子2
    :return: 返回中英文句子切分后的向量
    """
    # 把句子按字分开，中文按字分，英文按单词，数字按空格
    regEx = re.compile('[\\W]*')
    res = re.compile(r"([\u4e00-\u9fa5])")

    p1 = regEx.split(s1.lower())
    str1_list = []
    for str in p1:
        if res.split(str) == None:
            str1_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str1_list.append(ch)

    p2 = regEx.split(s2.lower())
    str2_list = []
    for str in p2:
        if res.split(str) == None:
            str2_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str2_list.append(ch)

    list_word1 = [w for w in str1_list if len(w.strip()) > 0]  # 去掉为空的字符
    list_word2 = [w for w in str2_list if len(w.strip()) > 0]  # 去掉为空的字符

    # 列出所有的词,取并集
    key_word = list(set(list_word1 + list_word2))
    # 给定形状和类型的用0填充的矩阵存储向量
    word_vector1 = np.zeros(len(key_word))
    word_vector2 = np.zeros(len(key_word))

    # 计算词频
    # 依次确定向量的每个位置的值
    for i in range(len(key_word)):
        # 遍历key_word中每个词在句子中的出现次数
        for j in range(len(list_word1)):
            if key_word[i] == list_word1[j]:
                word_vector1[i] += 1
        for k in range(len(list_word2)):
            if key_word[i] == list_word2[k]:
                word_vector2[i] += 1

    # 返回向量
    return word_vector1, word_vector2


def cos_dist(vec1, vec2):
    """
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 返回两个向量的余弦相似度
    """
    dist = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return dist


def calculate_similarity2(key_text, total_texts):
    sim = []
    for line in total_texts:
        vec1, vec2 = get_word_vector(key_text, line)
        dist = cos_dist(vec1, vec2)
        sim.append(dist)
    return sim


if __name__ == '__main__':
    s1 = "请书记关注A市A4区58车贷案"
    s2 = "严惩A市58车贷特大集资诈骗案保护伞"
    # s2 = "请清理A市人民西路137号人行道附近的僵尸车"

    vec1, vec2 = get_word_vector(s1, s2)
    dist1 = cos_dist(vec1, vec2)
    print(dist1)
