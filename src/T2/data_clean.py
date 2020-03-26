import jieba
import jieba.analyse
import xlrd

jieba.load_userdict("data/address_words.txt")
jieba.load_userdict("data/sogou_dict.txt")
jieba.analyse.set_stop_words("data/stop_words.txt")
sheet = xlrd.open_workbook("data/messages2.xlsx").sheet_by_index(0)
for i in range(1, sheet.nrows):
    msg = sheet.cell(i, 2).value
    seg_list = [key_word[0] for key_word in jieba.analyse.extract_tags(msg, 50, withWeight=True)]
    print("jieba Mode: " + '/'.join(list(seg_list)))
