def match_word2vec(init_data):
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

    def get_vec_list(doc: list, model):
        """计算文档向量"""
        vectors = []
        ignore = ["\t", " ", "\n"]
        words = [word for word in doc if word not in ignore]
        # 所有词向量求和并除以词数量
        for word in words:
            try:
                vectors.append(model[word].tolist())
            except KeyError:
                continue
        return vectors


    # word2vec向量化
    def w2v(train_seg):
        for i in range(len(train_seg)):
            train_seg[i] = train_seg[i].split(" ")
        file = 'data/Tencent_AILab_ChineseEmbedding.txt' # 加载腾讯word2vec是最全的
        wv_model = KeyedVectors.load_word2vec_format(file, binary=False)
        infered_vectors_list = []
        for text in train_seg:
            vector = get_vec_list(text, wv_model)
            infered_vectors_list.append(vector)
        return infered_vectors_list

    read2 = init_data
    # read2 = np.load('data/T3_messages.npy', allow_pickle=True)

    messages = [item[0] for item in read2]
    labels = [item[1] for item in read2]

    content = w2v(messages)
    reply = w2v(labels)

    messages1 = [(vector, label) for vector, label in zip(content, reply)]
    print(len(messages1))
    print(type(messages1))

    # np.save('data/T3_full_vector.npy', messages1, allow_pickle=True) # 得到词向量
    return messages1
