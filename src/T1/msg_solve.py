import xlrd
import jieba
import jieba.analyse
import langid
import numpy as np


class Message(object):
    def __init__(self, msg_id, user, topic, msg_time, detail, primary_class):
        self.msg_id = msg_id
        self.user = user
        self.topic = topic
        self.msg_time = msg_time
        self.detail = detail
        self.primary_class = primary_class
        self.topic_key_words = []
        self.detail_key_words = []

    def strip(self, stopwords_file):
        jieba.analyse.set_stop_words(stopwords_file)
        for key_word in jieba.analyse.extract_tags(self.topic, 50, withWeight=True):
            if langid.classify(key_word[0])[0] == 'zh':
                self.topic_key_words.append(key_word[0])
        for key_word in jieba.analyse.extract_tags(self.detail, 50, withWeight=True):
            if langid.classify(key_word[0])[0] == 'zh':
                self.detail_key_words.append(key_word[0])

    def __str__(self):
        return self.topic_key_words


class Classification(object):
    def __init__(self, primary, secondary, third):
        self.primary = primary
        self.secondary = secondary
        self.third = third
        self.key_words = []

    def strip(self, stopwords_file):
        jieba.analyse.set_stop_words(stopwords_file)
        self.key_words = [key_word[0] for key_word in
                          jieba.analyse.extract_tags("{} {} {}".format(self.primary, self.secondary, self.third),
                                                     1000, withWeight=True)]
        # self.key_words = jieba.cut("{} {} {}".format(self.primary, self.secondary, self.third))


def read_data(file):
    sheet = xlrd.open_workbook(file).sheet_by_index(0)
    msg = {}
    for i in range(1, sheet.nrows):
        msg_id = sheet.cell(i, 0).value
        user = sheet.cell(i, 1).value
        topic = sheet.cell(i, 2).value
        msg_time = sheet.cell(i, 3).value
        detail = sheet.cell(i, 4).value
        primary_class = sheet.cell(i, 5).value

        obj = Message(msg_id, user, topic, msg_time, detail, primary_class)
        obj.strip("data/stop_words.txt")
        msg[i] = obj.topic_key_words + obj.detail_key_words
        print(msg[i])
    return msg


def split_classification(file):
    import synonyms
    sheet = xlrd.open_workbook(file).sheet_by_index(0)
    msg = {}
    for i in range(1, sheet.nrows):
        primary = sheet.cell(i, 0).value
        secondary = sheet.cell(i, 1).value
        third = sheet.cell(i, 2).value
        obj = Classification(primary, secondary, third)
        obj.strip("data/stop_words.txt")
        if msg.get(primary, None):
            msg[primary].extend(obj.key_words)
        else:
            msg[primary] = obj.key_words

    for key in msg.keys():
        tmp = []
        for word in list(set(msg[key])):
            nearby_words, prob = synonyms.nearby(word)
            for i in range(len(nearby_words)):
                if prob[i] < 0.7 or langid.classify(nearby_words[i])[0] != "zh":
                    continue
                tmp.append(nearby_words[i])
        # msg[key] = list(set(tmp))
        msg[key] = tmp
        print(msg[key])
    return msg


if __name__ == "__main__":
    np.save("message.npy", read_data("data/messages.xlsx"))
    np.save("classifications.npy", split_classification("data/classifications.xlsx"))
    # print(read_data("data/messages.xlsx"))
    # print(split_classification("data/classifications.xlsx"))
