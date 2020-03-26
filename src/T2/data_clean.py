# encoding=utf-8
import jieba

jieba.initialize()
jieba.set_dictionary("data/address_words.txt")
jieba.enable_paddle()  # 启动paddle模式。 0.40版之后开始支持，早期版本不支持
jieba.load_userdict("data/address_words.txt")
# jieba.load_userdict("data/sogou_dict.txt")
strs = ["A市经济学院体育学院变相强制实习", "在A市人才app上申请购房补贴为什么通不过", "希望西地省把抗癌药品纳入医保范围", "A5区劳动东路魅力之城小区底层餐馆油烟扰民"]
# for str in strs:
#     seg_list = jieba.cut(str)
#     print("/".join(seg_list))
# jieba.add_word("经济学院", freq=20000, tag=None)
# jieba.suggest_freq('A5区', True)
for str in strs:
    # seg_list = jieba.cut(str, use_paddle=True, HMM=False)  # 使用paddle模式
    seg_list = jieba.cut(str, HMM=True)
    print("Paddle Mode: " + '/'.join(list(seg_list)))

# import string
#
# base = ["县", "市", "区", "省", "路", "镇", "小区"]
#
# fo = open("data/address_words.txt", "w")
#
# for word in string.ascii_uppercase:
#     for k in base:
#         fo.write(word + k + " 10" + '\n')
#     for j in range(1, 21):
#         fo.write(word + str(j) + " 10" + '\n')
#         for k in base:
#             fo.write(word + str(j) + k + " 10" + '\n')