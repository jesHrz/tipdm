import datetime
import warnings

import jieba
import jieba.analyse
import numpy as np
import pandas as pd
import xlrd
import xlsxwriter
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


# 数据录入 Id:用户id , title:留言主题 , train_text_seg:文本分词
def load_data(file_in):
    from xlrd import xldate_as_tuple
    jieba.load_userdict("data/extract_dict/address_words.txt")
    jieba.load_userdict("data/extract_dict/platform_words.txt")
    jieba.load_userdict("data/extract_dict/sogou_dict.txt")
    jieba.load_userdict("data/extract_dict/housing_estate.txt")
    stopwords = [line.strip() for line in open("data/extract_dict/stop_words_aa.txt", encoding='UTF-8').readlines()]
    sheet = xlrd.open_workbook(file_in).sheet_by_index(0)
    Id = []
    uId = []
    title = []
    time = []
    content = []
    bad = []
    good = []
    train_text_seg = list()

    for i in range(1, sheet.nrows):
        msg = sheet.cell(i, 2).value.encode('utf-8').decode('utf-8-sig')
        seg_list = jieba.cut(msg, cut_all=False, HMM=True)
        s = ''
        for word in seg_list:
            if word not in stopwords:
                if word != '\t':
                    if word != '\n':
                        s += word
                        s += " "

        train_text_seg.append(s)
        Id.append(sheet.cell(i, 0).value)
        uId.append(sheet.cell(i, 1).value.encode('utf-8').decode('utf-8-sig'))
        title.append(sheet.cell(i, 2).value.encode('utf-8').decode('utf-8-sig'))
        time.append(sheet.cell(i, 3).value)
        content.append(sheet.cell(i, 4).value.encode('utf-8').decode('utf-8-sig'))
        bad.append(sheet.cell(i, 5).value)
        good.append(sheet.cell(i, 6).value)

    return Id, uId, title, time, content, bad, good, train_text_seg


# tfidf向量化
def tfidf(train_seg):
    tfidf_vec = TfidfVectorizer()
    X = tfidf_vec.fit_transform(train_seg)
    return X


# 调参函数 X是文本向量
def search(X):
    # 构建空列表，用于保存不同参数组合下的结果
    res = []
    # 迭代不同的eps值
    for eps in np.arange(0.001, 0.7, 0.02):  # 需要修改调参
        # 迭代不同的min_samples值
        for min_samples in range(5, 21):  # 需要修改调参
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            # 模型拟合
            dbscan.fit(X)
            # 统计各参数组合下的聚类个数（-1表示异常点）
            n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
            # 异常点的个数
            outliners = np.sum(np.where(dbscan.labels_ == -1, 1, 0))
            # 统计每个簇的样本个数
            stats = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values)
            res.append({'eps': eps, 'min_samples': min_samples, 'n_clusters': n_clusters, 'outliners': outliners,
                        'stats': stats})
    # 将迭代后的结果存储到数据框中
    df = pd.DataFrame(res)

    # 根据条件筛选合理的参数组合
    print(df.loc[df.n_clusters == 13, :])  # 输出5类的情况


def draw(X, labels):
    from matplotlib import pyplot as plt
    from sklearn.decomposition import PCA

    X = PCA(n_components=2).fit_transform(X)
    labels_color_map = {
        0: '#20b2aa', 1: '#f20736', 2: '#FF7F24', 3: '#005073', 4: '#4d0404',
        5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
    }

    # ax = plt.subplot(111, projection='3d')
    ax = plt.subplot(111)
    for index, instance in enumerate(X):
        # print instance, index, labels[index]
        pca_comp_1, pca_comp_2 = X[index]
        color = labels_color_map[labels[index]]
        ax.scatter(pca_comp_1, pca_comp_2, c=color)
    plt.show()


# 聚类算法
def dbscan(eps, min_s, X):
    db = DBSCAN(eps=eps, min_samples=min_s, metric='cosine').fit(X)  # DBSCAN聚类方法 还有参数，matric = ""距离计算方法
    labels = db.labels_  # 和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目

    print('分簇的数目: %d' % n_clusters_)  # 输出每一簇的元素数
    X_ = []
    labels_ = []
    for i in range(n_clusters_):
        print('簇 ', i, '的个数:', end='')
        cont = 0
        for j in range(len(labels)):
            if labels[j] == i:
                cont += 1
                X_.append(X[j].toarray()[0])
                labels_.append(i)
        print(cont)
    # draw(X_, labels_)
    return labels


def clustering(file_in, file_out):
    Id, uId, title, time, content, bad, good, train_seg = load_data(file_in)

    tic = datetime.datetime.now()  # 计时
    X = tfidf(train_seg)
    toc = datetime.datetime.now()
    print("vectorized finished in {}".format(toc - tic))

    # tc = 0
    # if tc == 1:
    #     search(X)  # 调参，需要自行设置范围
    # # 选最优参数跑
    # else:
    labels = dbscan(0.561, 8, X)

    workbook3 = xlsxwriter.Workbook(file_out)  # 存放结果
    worksheet3 = workbook3.add_worksheet(u'sheet1')
    col_name = ["留言编号", "留言用户", "留言主题", "留言时间", "留言详情", "反对数", "点赞数", "类"]
    for i, name in enumerate(col_name):
        worksheet3.write(0, i, name)
    for i in range(len(labels)):
        worksheet3.write(i + 1, 0, Id[i])  # 问题ID
        worksheet3.write(i + 1, 1, uId[i])  # 用户ID
        worksheet3.write(i + 1, 2, title[i])  # 主题
        worksheet3.write(i + 1, 3, time[i])  # 时间
        worksheet3.write(i + 1, 4, content[i])  # 内容
        worksheet3.write(i + 1, 5, bad[i])  # 反对
        worksheet3.write(i + 1, 6, good[i])  # 点赞

        worksheet3.write(i + 1, 7, labels[i])  # 类别

    workbook3.close()


if __name__ == '__main__':
    Id, uId, title, time, content, bad, good, train_seg = load_data('data/messages/附件3_clean.xlsx')
    X = tfidf(train_seg)
    labels = dbscan(0.561, 8, X)