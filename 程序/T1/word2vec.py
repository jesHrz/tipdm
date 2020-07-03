import numpy as np
from gensim.models import KeyedVectors

# 定义通过word2vec计算文档向量的方法
def get_doc_vec(doc: list, model):
    """计算文档向量"""
    ignore = ["\t", " ", "\n"]
    words = [word for word in doc if word not in ignore]
    # 所有词向量求和并除以词数量
    words_num = len(words)

    vec_sum = np.zeros(200)
    for word in words:
        try:
            vec_sum += model[word]
        except KeyError:
            words_num -= 1
            continue
    if words_num == 0:
        return vec_sum
    else:
        return vec_sum / words_num

# word2vec向量化
def w2v(train_seg):
    for i in range(len(train_seg)):
        train_seg[i] = train_seg[i].split(" ")
    file = 'data/Tencent_AILab_ChineseEmbedding.txt' #加载腾讯word2vec是最全的
    wv_model = KeyedVectors.load_word2vec_format(file, binary=False)
    infered_vectors_list = []
    for text in train_seg:
        vector = get_doc_vec(text, wv_model).tolist()
        infered_vectors_list.append(vector)
    return infered_vectors_list


read2 = np.load('data/messages_new.npy', allow_pickle=True)

messages = [item[0] for item in read2]
labels = [item[1] for item in read2]
vector_list = w2v(messages)

messages1 = [(vector, label) for vector, label in zip(vector_list, labels)]
print(len(messages1))
print(type(messages1))

np.save('data/messages_w2v.npy', messages1, allow_pickle=True)


read2 = np.load('data/all_messages_new.npy', allow_pickle=True)

messages = [item[0] for item in read2]
labels = [item[1] for item in read2]
vector_list = w2v(messages)

messages1 = [(vector, label) for vector, label in zip(vector_list, labels)]
print(len(messages1))
print(type(messages1))

np.save('data/all_messages_w2v.npy', messages1, allow_pickle=True)