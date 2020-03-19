import jieba
from gensim.similarities import SparseMatrixSimilarity
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np
import xlrd
from copy import deepcopy as dco
import xlsxwriter


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

message=np.load('message.npy').item()
classifications=np.load('classifications.npy').item()

template_cnt = {
    '城乡建设': 0,
    '党务政务': 0,
    '国土资源': 0,
    '环境保护': 0,
    '纪检监察': 0,
    '交通运输': 0,
    '经济管理': 0,
    '科技与信息产业': 0,
    '民政': 0,
    '农村农业': 0,
    '商贸旅游': 0,
    '卫生计生': 0,
    '政法': 0,
    '教育文体': 0,
    '劳动和社会保障': 0}

# print(message)
# print(classifications)

keys = list(message.values())
ans = list(classifications)
# 将字典的value转换成列表
sent = list(classifications.values())
list_mine = []

with open('mine.txt','w') as f:
    for k in keys:
        grade = similar_result(sent,k)
        list_mine.append(ans[np.argsort(grade)[-1]])

path = '附件2.xlsx'
ExcelFile1=xlrd.open_workbook(path)
sheet1=ExcelFile1.sheet_by_index(0)
list_ans=[]
# workbook3 = xlsxwriter.Workbook('wrong.xlsx')  # 创建一个excel文件
# worksheet3 = workbook3.add_worksheet(u'sheet1')  # 在文件中创建一个名为TEST的sheet,不加名字默认为sheet1
for i in range(1, sheet1.nrows):

        text = sheet1.cell(i, 5).value.encode('utf-8').decode('utf-8-sig')
        list_ans.append(text)
#         if list_mine[i-1] != text:
#             worksheet3.write(i, 0, list_mine[i-1])
# workbook3.close()



#evalue
true_cnt = dco(template_cnt)
ans_cnt = dco(template_cnt)
mine_cnt = dco(template_cnt)


if len(list_ans) != len(list_mine):
    print("error")

for pos in range(0, len(list_ans)):
    ans_cnt[list_ans[pos]] += 1
    mine_cnt[list_mine[pos]] += 1
    if list_ans[pos] == list_mine[pos]:
        true_cnt[list_ans[pos]] += 1

ans, num = 0, 0
keys = ans_cnt.keys()

for item in keys:
    if ans_cnt[item] != 0:
        num += 1
        if mine_cnt[item] == 0:
            pi = 0
        else:
            pi = true_cnt[item] / mine_cnt[item]
        ri = true_cnt[item] / ans_cnt[item]
        ans += 2 * pi * ri / (pi + ri)
        print(item,':',2 * pi * ri / (pi + ri))
ans /= num
print("总分:",ans)

