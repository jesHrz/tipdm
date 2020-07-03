import jieba
import numpy as np
import xlrd


stopwords = [line.strip() for line in open('data/stop_words.txt',encoding='UTF-8').readlines()]
sheet = xlrd.open_workbook("data/附件2.xlsx").sheet_by_index(0)
messages = []
labels = []

for i in range(1, sheet.nrows):
    msg = sheet.cell(i, 2).value.encode('utf-8').decode('utf-8-sig') +sheet.cell(i, 4).value.encode('utf-8').decode('utf-8-sig')
    label = sheet.cell(i, 5).value.encode('utf-8').decode('utf-8-sig')
    seg_list = jieba.cut(msg,cut_all=False,HMM=True)
    s = ''
    for word in seg_list:
        if word not in stopwords:
            if word != '\t':
                if word != '\n':
                    s += word
                    s += " "
    messages.append(s)
    labels.append(label)

messages1 = [(message, label) for message, label in zip(messages, labels)]
print(len(messages1))
print(type(messages1))

np.save('data/messages_new.npy', messages1, allow_pickle=True)


sheet = xlrd.open_workbook("data/bxhs_more.xlsx").sheet_by_index(0)

for i in range(1, sheet.nrows):
    msg = sheet.cell(i, 0).value.encode('utf-8').decode('utf-8-sig')
    label = sheet.cell(i, 1).value.encode('utf-8').decode('utf-8-sig')
    seg_list = msg.split()
    s = ''
    for word in seg_list:
        if word not in stopwords:
            if word != '\t':
                if word != '\n':
                    s += word
                    s += " "

    messages.append(s)
    labels.append(label)


messages2 = [(message, label) for message, label in zip(messages, labels)]
print(len(messages2))
print(type(messages2))

np.save('data/all_messages_new.npy', messages2, allow_pickle=True)

