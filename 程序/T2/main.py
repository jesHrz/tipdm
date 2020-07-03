import os

import xlrd
import xlsxwriter
import re
import time
from textrank4zh import TextRank4Sentence

import cluster
import hot_points

# 获取项目路径
current_folder_path = os.path.dirname(os.path.realpath(__file__))
# 获取数据存放目录路径
data_path = os.path.join(current_folder_path, 'data')
extra_dict_path = os.path.join(data_path, 'extra_dict')
messages_path = os.path.join(data_path, 'messages')
results_path = os.path.join(data_path, 'results')


def get_abstract(texts: list) -> str:
    """
    利用textrank算法, 获得文本摘要
    :param texts: list, 原文本
    :return: str, 文本摘要
    """
    text = '\n'.join(texts)
    tr4s = TextRank4Sentence(delimiters='\n')
    tr4s.analyze(text=text, lower=True, source='all_filters')
    abstract = tr4s.get_key_sentences(num=1)[0]['sentence']
    return abstract


def filter_data(file_in: str, file_out: str = None, iteration=7, delta=0.005, max_label=13) -> dict:
    """
    对合并后的xlsx文件去除噪声
    :param file_in: 原文件
    :param file_out: 过滤噪声后的文件
    :param iteration: 迭代次数
    :param delta: 每次迭代增加的相似度
    :param max_label: 区分标签
    :return: 标签到留言集合的dict
    """
    import xlrd
    sheet = xlrd.open_workbook(file_in).sheet_by_index(0)
    messages = list()
    dict1 = dict()
    for i in range(1, sheet.nrows):
        cells = sheet.row_values(i)
        messages.append(cells)
        label = int(float(cells[7]))
        if label != -1:
            if label not in dict1:
                dict1[label] = []
            dict1[label].append(i - 1)
    for k in range(iteration):
        for label, values in dict1.items():
            try:
                titles = [messages[value][2] for value in values]
                if label <= max_label:
                    abstract = get_abstract(titles)
                else:
                    mxi = values[0]
                    for value in values:
                        if messages[value][6] > messages[mxi][6]:
                            mxi = value
                    abstract = messages[mxi][2]
                similarity = hot_points.calculate_similarity2(key_text=abstract, total_texts=titles)

                for i, sim in enumerate(similarity):
                    print('iter %d label %d key_text %s 与 text %s 相似度为：%f' % (k, label, abstract, titles[i], sim))

                filtered = [values[i] for i, sim in enumerate(similarity) if sim > 0.01 + delta * k]
                dict1[label] = filtered

            except Exception as e:
                print(e)
                continue

    for label, values in dict1.items():
        dict1[label] = [messages[value] for value in values]

    if file_out:
        col_name = ['留言编号', '留言用户', '留言主题', '留言时间', '留言详情', '反对数', '点赞数', '类']
        workbook = xlsxwriter.Workbook(file_out)
        sheet = workbook.add_worksheet()
        for i, name in enumerate(col_name):
            sheet.write(0, i, name)
        rows = 1
        for label, values in dict1.items():
            for value in values:
                for j in range(8):
                    sheet.write(rows, j, value[j])
                rows += 1
        workbook.close()
    return dict1


def get_num_of_value_no_repeat(list1):
    """
    获取列表中不重复的值的个数
    :param list1: 列表
    :return: int，列表中不重复的值的个数
    """
    num = len(set(list1))
    return num


def get_score(file: str) -> dict:
    import xlrd
    import numpy as np

    def min_max_scale(nums):
        mx = np.max(nums)
        mn = np.min(nums)
        for k, value in enumerate(nums):
            nums[k] = (value - mn) / (mx - mn)
        return nums

    def softmax_scale(nums):
        s = sum([np.exp(num) for num in nums])
        for k, value in enumerate(nums):
            nums[k] = np.exp(value) / s
        return nums

    def z_score_scale(nums):
        mean = np.mean(nums)
        sigma = np.std(nums)
        for k, value in enumerate(nums):
            nums[k] = (value - mean) / sigma
        return nums

    def wilson_score(pos, neg, p_z=2.0, k=0):
        """
        威尔逊得分计算函数
        :param pos: 正例数
        :param neg: 负例数
        :param p_z: 正太分布的分位数
        :param k: 光顺, 避免除0
        :return: 威尔逊得分
        """
        total = pos + neg + k
        pos_rat = (pos + 1) / total  # 正例比率
        score = (pos_rat + (np.square(p_z) / (2. * total))
                 - ((p_z / (2. * total)) * np.sqrt(4. * total * (1. - pos_rat) * pos_rat + np.square(p_z)))) / \
                (1. + np.square(p_z) / total)
        return score

    sheet = xlrd.open_workbook(file).sheet_by_index(0)
    dict1 = dict()
    dict_len = dict()
    dict_pos = dict()
    dict_neg = dict()
    for i in range(1, sheet.nrows):
        user_id = sheet.cell(i, 1).value
        neg_num = sheet.cell(i, 5).value
        pos_num = sheet.cell(i, 6).value
        label = int(float(sheet.cell(i, 7).value))
        if label != -1:
            if label not in dict1:
                dict1[label] = []
            dict1[label].append((user_id, pos_num, neg_num))
    for key, values in dict1.items():
        dict_len[key] = get_num_of_value_no_repeat([value[0] for value in values])
        dict_pos[key] = sum([int(float(value[1])) for value in values])
        dict_neg[key] = sum([int(float(value[2])) for value in values])

    # dict_len = dict(zip(list(dict_len.keys()), min_max_scale(list(dict_len.values()))))  # 选择不同的归一化函数可能得到不同的结果
    # dict_len = dict(zip(list(dict_len.keys()), softmax_scale(list(dict_len.values()))))  # 选择不同的归一化函数可能得到不同的结果
    dict_len = dict(zip(list(dict_len.keys()), z_score_scale(list(dict_len.values()))))  # 选择不同的归一化函数可能得到不同的结果

    dict_ans = dict()
    for key in dict_len:
        # print(key, dict_len[key], wilson_score(dict_pos[key], dict_neg[key], k=10))
        dict_ans[key] = dict_len[key] + wilson_score(dict_pos[key], dict_neg[key], k=50)  # k可以设置光顺值,避免除0的尴尬场面
    return dict_ans


def merge_xlsx(file1, file2, file_to):
    sheet1 = xlrd.open_workbook(file1).sheet_by_index(0)
    sheet2 = xlrd.open_workbook(file2).sheet_by_index(0)
    workbook = xlsxwriter.Workbook(file_to)
    sheet3 = workbook.add_worksheet()
    col_name = ["留言编号", "留言用户", "留言主题", "留言时间", "留言详情", "反对数", "点赞数", "类"]
    for i, name in enumerate(col_name):
        sheet3.write(0, i, name)
    rows = 1
    max_label = 0
    for i in range(1, sheet1.nrows):
        cells = sheet1.row_values(i)
        label = int(cells[7])
        if label == -1: continue
        for j, cell in enumerate(cells):
            sheet3.write(rows, j, str(cell))
        rows += 1
        max_label = max(max_label, label)
    for i in range(1, sheet2.nrows):
        cells = sheet2.row_values(i)
        cells[7] = str(int(cells[7]) + max_label + 1)
        for j, cell in enumerate(cells):
            sheet3.write(rows, j, str(cell))
        rows += 1
    workbook.close()
    return max_label


def generate_hotpoints_xlsx(messages_dict, scores):
    def get_date_range(messages):
        dates = [message[3] for message in messages]
        dates_format = list()
        regex = re.compile('[0-9]{4}/[0-9]+/[0-9]+')
        for date in dates:
            for matched in regex.findall(date.strip()):
                dates_format.append(time.strptime(matched, "%Y/%m/%d"))
        dates_format.sort()
        begin = time.strftime("%Y/%m/%d", dates_format[0])
        end = time.strftime("%Y/%m/%d", dates_format[-1])
        return begin + "-" + end

    def get_ref(messages):
        return ""

    col_name1 = ['热度排名', '问题ID', '热度指数', '时间范围', '地点/人群', '问题描述']
    col_name2 = ['问题ID', '留言编号', '留言用户', '留言主题', '留言时间', '留言详情', '点赞数', '反对数']
    workbook1 = xlsxwriter.Workbook(os.path.join(results_path, '热点问题表.xlsx'))
    workbook2 = xlsxwriter.Workbook(os.path.join(results_path, '热点问题留言明细表.xlsx'))
    sheet1 = workbook1.add_worksheet()
    sheet2 = workbook2.add_worksheet()
    for i, name in enumerate(col_name1):
        sheet1.write(0, i, name)
    for i, name in enumerate(col_name2):
        sheet2.write(0, i, name)
    rows1 = 1
    rows2 = 1
    for i, item in enumerate(scores):
        if i >= 5: break
        label, score = item
        rank = i + 1
        pid = i + 1
        date_range = get_date_range(messages_dict[label])
        ref = get_ref(messages_dict[label])
        abstract = get_abstract([message[2] for message in messages_dict[label]])
        sheet1.write(rows1, 0, rank)
        sheet1.write(rows1, 1, pid)
        sheet1.write(rows1, 2, score)
        sheet1.write(rows1, 3, date_range)
        sheet1.write(rows1, 4, ref)
        sheet1.write(rows1, 5, abstract)
        rows1 += 1

        for message in messages_dict[label]:
            sheet2.write(rows2, 0, str(pid))
            sheet2.write(rows2, 1, str(int(float(message[0]))))
            sheet2.write(rows2, 2, message[1])
            sheet2.write(rows2, 3, message[2])
            sheet2.write(rows2, 4, message[3])
            sheet2.write(rows2, 5, message[4])
            sheet2.write(rows2, 6, str(int(float(message[6]))))
            sheet2.write(rows2, 7, str(int(float(message[5]))))
            rows2 += 1
    workbook1.close()
    workbook2.close()


def main():
    cluster.clustering(os.path.join(messages_path, '附件3_clean.xlsx'), os.path.join(results_path, 'result_last.xlsx'))
    hot_points.hotpoints_classify(os.path.join(results_path, 'result_last.xlsx'),
                                  os.path.join(results_path, 'result_hotpoints.xlsx'))

    max_label = merge_xlsx(os.path.join(results_path, 'result_last.xlsx'),
                           os.path.join(results_path, 'result_hotpoints.xlsx'),
                           os.path.join(results_path, 'result_merge.xlsx'))

    messages_dcit = filter_data(os.path.join(results_path, 'result_merge.xlsx'),
                                os.path.join(results_path, 'result_filter.xlsx'), max_label=max_label)

    ans = get_score(os.path.join(results_path, 'result_filter.xlsx'))
    score = sorted(ans.items(), key=lambda x: x[1], reverse=True)

    print(score)
    generate_hotpoints_xlsx(messages_dcit, score)


if __name__ == "__main__":
    main()