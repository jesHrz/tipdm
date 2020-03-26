import jieba
import xlrd

jieba.load_userdict("data/address_words.txt")
sheet = xlrd.open_workbook("data/messages2.xlsx").sheet_by_index(0)
for i in range(1, sheet.nrows):
    msg = sheet.cell(i, 2).value
    seg_list = jieba.cut(msg, HMM=True)
    print("jieba Mode: " + '/'.join(list(seg_list)))
