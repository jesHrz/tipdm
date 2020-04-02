import jieba
import jieba.analyse
import xlrd
import string
from src.util.analyse import striper

address_list = ["街道", "学校", "学院", "平台", "小区", "省", "市", "县", "区", "村", "路"]
platform_list = open("data/platform_words.txt", "r+").read().split('\n')


def judge_suffix(s):
    for suffix in address_list:
        if suffix == "平台" and s in platform_list:
            return suffix
        if s == suffix:
            return None
        if len(s) < len(suffix):
            continue
        if s[-len(suffix):] == suffix:
            return suffix
    return None


def data_analyse():
    jieba.load_userdict("data/address_words.txt")
    jieba.load_userdict("data/platform_words.txt")
    jieba.load_userdict("data/sogou_dict.txt")
    jieba.analyse.set_stop_words("data/stop_words.txt")
    sheet = xlrd.open_workbook("data/messages2.xlsx").sheet_by_index(0)

    for i in range(1, sheet.nrows):
        msg = sheet.cell(i, 2).value.strip() + " " + sheet.cell(i, 4).value.strip()
        seg_list = striper.strip(msg)
        print("jieba Mode: " + '/'.join(list(seg_list)))
        ans_dict = dict()
        index = 0
        for seg in seg_list:
            suffix = judge_suffix(seg)
            if suffix and len(ans_dict.get(suffix, ("", 0))[0]) < len(seg):
                ans_dict[suffix] = (seg, index)
            index += 1
        address_result = "".join([address[0] for address in sorted(ans_dict.values(), key=lambda t: t[1])])
        print(i, "地名：" + address_result)


def main():
    data_analyse()


if __name__ == "__main__":
    main()
