import xlrd
import xlsxwriter
from similarity import calculate_similarity, calculate_similarity2


def data_acquire(path):
    result = []
    sheet = xlrd.open_workbook(path).sheet_by_index(0)
    for line in range(1, sheet.nrows):
        tmp = []
        for j in range(0, 8):
            tmp.append(sheet.cell(line, j).value)
        # 只取出 -1 的类
        if tmp[7] == -1:
            result.append(tmp)
    return result


def hotpoints_acquire(total_data):
    result, tmp = [], []
    for i in range(0, len(total_data)):
        line = total_data[i]
        arr = {"id": i, "value": -line[5] + line[6]}
        tmp.append(arr)
    tmp.sort(key=lambda x: x["value"])
    tmp.reverse()

    # 获得最大热点
    total_data[tmp[0]["id"]][7] = -2
    result = total_data[tmp[0]["id"]]

    # 从原数据集中删去赞数最多的数据
    new_data = []
    for line in total_data:
        if line[7] != -2:
            new_data.append(line)
    return result, new_data


def cluster_acquire(total_data, ratio):
    hotpoints, total_data = hotpoints_acquire(total_data)
    total_texts, cluster = [], []
    key_text = hotpoints[2]
    for line in total_data:
        total_texts.append(line[2])

    sim = calculate_similarity(key_text, total_texts)
    result = []
    for id in range(0, len(sim)):
        arr = {"id": id, "value": sim[id]}
        result.append(arr)
    result.sort(key=lambda x: x["value"])
    result.reverse()
    # 选取相似度 > 0.1 的类
    for line in result:
        if line["value"] > ratio:
            cluster.append(total_data[line["id"]])
            total_data[line["id"]][7] = -2
        else:
            break
    # 更新数据集
    new_data = []
    for line in total_data:
        if line[7] != -2:
            new_data.append(line)
    return hotpoints, cluster, new_data


def write_excel_xls(path, sheet_name, value):
    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet(sheet_name)

    row_name = ["留言编号", "留言用户", "留言主题", "留言时间", "留言详情", "反对数", "点赞数", "类"]
    row_num = len(value) + 1
    col_num = len(value[0])

    for col in range(0, col_num):
        worksheet.write(0, col, row_name[col])
    for row in range(1, row_num):
        for col in range(0, col_num):
            worksheet.write(row, col, value[row - 1][col])
    workbook.close()


def hotpoints_merge(total_texts):
    tmp, result, total_data = [], [], []
    for i in range(0, len(total_texts)):
        line = total_texts[i]
        arr = {"id": i, "value": -line[5] + line[6]}
        tmp.append(arr)
    tmp.sort(key=lambda x: x["value"], reverse=True)
    for i in range(0, len(tmp)):
        if total_texts[tmp[i]["id"]][6] > 50 and i < 2:
            result.append(total_texts[tmp[i]["id"]])
        else:
            total_data.append(total_texts[tmp[i]["id"]])
    return result, total_data


def cluster_filter(key_text, total_texts, ratio, flag):
    result, total_texts = hotpoints_merge(total_texts)
    tmp = []
    for line in total_texts:
        tmp.append(line[2])
    if flag == 1:
        sim = calculate_similarity(key_text[2], tmp)
    else:
        sim = calculate_similarity2(key_text[2], tmp)

    for index in range(0, len(total_texts)):
        # print('key_text 与 text%d 相似度为：%.2f' % (index + 1, sim[index]), tmp[index])
        if sim[index] > ratio:
            result.append(total_texts[index])
    return result


def hotpoints_classify(path1, path2):
    total_data = data_acquire(path1)
    final_result = []
    for i in range(0, 10):
        hotpoints, cluster, total_data = cluster_acquire(total_data, 0.1)
        cluster = cluster_filter(hotpoints, cluster, 0.1, 1)
        # print("------------------------")
        cluster = cluster_filter(hotpoints, cluster, 0.42, 2)
        cluster.append(hotpoints)
        for line in cluster:
            line[7] = i
            final_result.append(line)
        # print(hotpoints[0], hotpoints[2], hotpoints[5], hotpoints[6], hotpoints[7], len(cluster))
        # for line in cluster:
        #     print(line[0], line[2], line[5], line[6], line[7])
        # print("\n")
    write_excel_xls(path2, "sheet1", final_result)


if __name__ == '__main__':
    hotpoints_classify("data/result_last.xlsx", "data/result_hotpoints.xlsx")