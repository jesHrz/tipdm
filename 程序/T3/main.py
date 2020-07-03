from match_score import match_score
from smooth_score import smooth_score
from time_score import time_score

def read_excel_xls(path):
    import xlrd
    result = []
    sheet = xlrd.open_workbook(path).sheet_by_index(0)
    for line in range(1, sheet.nrows):
        tmp = []
        for j in range(0, 7):
            tmp.append(sheet.cell(line, j).value)
        result.append(tmp)
    return result

def write_excel_xls(path, sheet_name, value):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet(sheet_name)

    row_name = ["留言编号","留言用户","留言主题","留言时间","留言详情","答复意见","答复时间","匹配度","语法通顺度","语义通顺度","通顺度","及时度","评价总分"]
    row_num = len(value)+1
    col_num = len(value[0])
    
    for col in range(0, col_num):
        worksheet.write(0, col, row_name[col])
    for row in range(1, row_num):
        for col in range(0, col_num):
            worksheet.write(row, col, value[row-1][col])
    workbook.close()

def calc_final_result(data):
    matchscore = match_score()
    smoothscore = smooth_score()
    timescore = time_score()
    for index in range(0, len(data)):
        a1, a2, a3 = 0, 0, 0
        if matchscore[index] >= 0.3:
            a1, a2, a3 = 0.5, 0.3, 0.2
        else:
            a1 = 0.5 + 0.5 * (0.3 - matchscore[8]) / 0.3
            a2 = (1 - a1) * 3 / 5
            a3 = 1 - a1 - a2
        data.append(matchscore[index])
        data.append(smoothscore[index][0])
        data.append(smoothscore[index][1])
        data.append(smoothscore[index][2])
        data.append(timescore[index])
        
        totalscore = (matchscore[index] * a1 + smoothscore[index][2] * a2 + time_score[index] * a3) * 100
        data.append(totalscore)

    return data

if __name__ == "__main__":
    load_path = "./data/附件4.xlsx"
    save_path = "./data/留言评价结果.xlsx"
    data = read_excel_xls(load_path)
    data = calc_final_result(data)
    write_excel_xls(save_path, "留言评价结果", data)