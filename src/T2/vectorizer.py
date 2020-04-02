# from gensim.models import word2vec
# from gensim.models import KeyedVectors
#
# # 设置词语向量维度
# num_featrues = 300
# # 保证被考虑词语的最低频度
# min_word_count = 0
# # 设置并行化训练使用CPU计算核心数量
# num_workers = 2
# # 设置词语上下午窗口大小
# context = 5
# downsampling = 1e-3
#
# model_file = "data/model.bin"
#
#
# def train(text_seg):
#     try:
#         model = word2vec.Word2Vec.load(model_file)
#     except FileNotFoundError as e:
#         with open("data/seg_list.txt", "w") as f:
#             f.write(" ".join(text_seg))
#         sentences = word2vec.LineSentence("data/seg_list.txt")
#         model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=100)
#         model.save(model_file)
#     return model
#
#
# def load_vector(file):
#     wv_from_text = KeyedVectors.load_word2vec_format(file, binary=False)  # 加载时间比较长
#     wv_from_text.init_sims(replace=True)
#     return wv_from_text

from gensim.models import doc2vec
from sklearn.cluster import KMeans


def train_doc2vec(seg_lists):
    tag_tokenized = [doc2vec.TaggedDocument(seg_lists[i], [i]) for i in range(len(seg_lists))]
    model = doc2vec.Doc2Vec(size=50, min_count=2)
    model.build_vocab(tag_tokenized)
    model.train(tag_tokenized, total_examples=model.corpus_count, epochs=model.iter)
    # 保存模型
    model.save('data/doc2vec.model')
    return model


def cluster(seg_lists):
    def scale(dt):
        sigma = sum(dt)
        mu = sigma / len(dt)
        return [(val - mu) / sigma for val in dt]

    seg_lists = [doc2vec.TaggedDocument(seg_lists[i], [i]) for i in range(len(seg_lists))]
    infered_vectors_list = []
    print("load doc2vec model...")
    model_dm = doc2vec.Doc2Vec.load('data/doc2vec.model')
    print("load train vectors...")
    for text, label in seg_lists:
        vector = model_dm.infer_vector(text)
        # vector = scale(vector)
        infered_vectors_list.append(vector)
    # import numpy as np
    # infered_vectors_list = np.load("data/vec.npy", allow_pickle=True)
    # np.save("data/doc2vec.txt", np.array(infered_vectors_list))
    for clu in range(20, 21):
        print("train k-mean model ...", "k=", clu)
        kmean_model = KMeans(n_clusters=clu, n_jobs=-1)
        # kmean_model.fit(infered_vectors_list)
        labels = kmean_model.fit_predict(infered_vectors_list)
        cluster_centers = kmean_model.cluster_centers_
        ret = dict()
        for i in range(len(seg_lists)):
            if labels[i] in ret:
                ret[labels[i]].append(i)
            else:
                ret[labels[i]] = [i]
        print("inertia: {}".format(kmean_model.inertia_))
        with open('data/cluster_result.txt', 'w', encoding='utf-8') as out_f:
            out_f.write("k={}\tinertia={}\n".format(clu, kmean_model.inertia_))
            for key in ret:
                out_f.write("%d:\t" % key)
                for index in ret[key]:
                    out_f.write(" %d" % index)
                out_f.write("\n")
        print(ret)