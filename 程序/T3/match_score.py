from word2vec import match_word2vec
from match_sim import mathch_sim

def match_init():
    import warnings
    warnings.filterwarnings("ignore")
    import jieba
    import jieba.analyse
    import xlrd
    import numpy as np

    jieba.load_userdict("data/地址词.txt") # 地址数据
    stopwords = [line.strip() for line in open('data/停用词.txt',encoding='UTF-8').readlines()] # 停用词
    sheet = xlrd.open_workbook("data/附件4.xlsx").sheet_by_index(0) # T3初始数据
    content = []
    reply = []

    for i in range(1, sheet.nrows):
        msg = sheet.cell(i, 2).value.encode('utf-8').decode('utf-8-sig') + sheet.cell(i, 4).value.encode('utf-8').decode('utf-8-sig')
        ans = sheet.cell(i, 5).value.encode('utf-8').decode('utf-8-sig')
        seg_list = jieba.cut(msg,cut_all=False,HMM=True)
        s=''
        for word in seg_list:
            if word not in stopwords:
                if word != '\t':
                    if word != '\n':
                        s += word
                        s += " "
        content.append(s)
        seg_list2 = jieba.cut(ans, cut_all=False, HMM=True)
        ss = ''
        for word in seg_list2:
            if word not in stopwords:
                if word != '\t':
                    if word != '\n':
                        ss += word
                        ss += " "
        reply.append(ss)

    messages1 = [(vector, label) for vector, label in zip(content, reply)]
    print(len(messages1))
    print(type(messages1))

    # np.save('data/T3_messages.npy', messages1, allow_pickle=True) # 保存数据
    return messages1

def match_score():
    init_data =  match_init()
    word_vector = match_word2vec(init_data)
    result = mathch_sim(word_vector)
    return result
