def time_score():
    import re
    import xlrd
    from chinese_calendar import is_workday
    from datetime import datetime, timedelta
    from xlrd import xldate_as_tuple

    def data_acquire(path):
        result = []
        sheet = xlrd.open_workbook(path).sheet_by_index(0)
        for line in range(1, sheet.nrows):
            tmp = []
            for j in range(0, 7):
                tmp.append(sheet.cell(line, j).value)
            result.append(tmp)
        return result

    def get_score(begin, end, x, alpha=1.1):
        begin = tuple(map(int, begin))
        end = tuple(map(int, end))
        now = datetime(*begin)
        end = datetime(*end)

        score = 100
        oneday = timedelta(days=1)
        penalties = [16, 28, 46, 70, 100]
        workday = 0
        while workday < penalties[-1] and now < end:
            if is_workday(now):
                workday += 1
                if workday >= 10:
                    factor = 1
                    for penalty in penalties:
                        if workday < penalty:
                            score -= factor * x
                            break
                        factor *= alpha
            now += oneday
        # print(workday)
        if workday >= penalties[-1]:
            score = 0
        return score

    data_load_path = "./data/附件4.xlsx"
    messages = data_acquire(data_load_path)

    regex = re.compile('([0-9]+)/([0-9]+)/([0-9]+) ?')
    date_scores = list()
    for i, message in enumerate(messages):
        if isinstance(message[3], float):
            begin = tuple(datetime(*xldate_as_tuple(message[3], 0)).strftime('%Y/%m/%d').split('/'))
        else:
            begin = regex.findall(message[3])[0]

        if isinstance(message[6], float):
            end = tuple(datetime(*xldate_as_tuple(message[6], 0)).strftime('%Y/%m/%d').split('/'))
        else:
            end = regex.findall(message[6])[0]
        score = get_score(begin, end, 0.82)
        date_scores.append(score)
        # print(i, begin, end)
        # print(get_score(begin, end, 0.82))
    max_score = max(date_scores)
    min_score = min(date_scores)
    for i, score in enumerate(date_scores):
        date_scores[i] = (score - min_score) / (max_score - min_score)
    return date_scores