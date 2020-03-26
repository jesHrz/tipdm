import jieba
import jieba.analyse
import xlrd
import string

address_list = ["国", "省", "市", "县", "区", "村", "街道", "学校", "学院"]


def judge_suffix(str_a, str_b):
    len1 = len(str_a)
    len2 = len(str_b)
    if len1 < len2:
        return False
    else:
        for pos in range(0, len2):
            if str_b[pos] != str_a[len1 - len2 + pos]:
                return False
    return True


def data_analyse():
    jieba.load_userdict("data/address_words.txt")
    jieba.load_userdict("data/sogou_dict.txt")
    jieba.analyse.set_stop_words("data/stop_words.txt")
    sheet = xlrd.open_workbook("data/messages2.xlsx").sheet_by_index(0)

    for i in range(1, sheet.nrows):
        msg = sheet.cell(i, 2).value + sheet.cell(i, 4).value
        seg_list = [key_word[0] for key_word in jieba.analyse.extract_tags(msg, 50, withWeight=True)]
        # print("jieba Mode: " + '/'.join(list(seg_list)))
        address_result = ""
        for seg in seg_list:
            for address in address_list:
                if judge_suffix(seg, address):
                    address_result += seg
        print("地名：" + address_result)


def main():
    data_analyse()


if __name__ == "__main__":
    main()
