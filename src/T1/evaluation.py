from copy import deepcopy as dco


class F1Score(object):
    template_cnt = {
        '城乡建设': 0,
        '党务政务': 0,
        '国土资源': 0,
        '环境保护': 0,
        '纪检监察': 0,
        '交通运输': 0,
        '经济管理': 0,
        '科技与信息产业': 0,
        '民政': 0,
        '农村农业': 0,
        '商贸旅游': 0,
        '卫生计生': 0,
        '政法': 0,
        '教育文体': 0,
        '劳动和社会保障': 0,
    }

    def evaluate(self, list_ans, list_mine):
        true_cnt = dco(self.template_cnt)
        ans_cnt = dco(self.template_cnt)
        mine_cnt = dco(self.template_cnt)
        keys = ans_cnt.keys()
        result, ans, num = {}, 0, 0

        if len(list_ans) != len(list_mine):
            print("error")
            exit(0)

        for pos in range(0, len(list_ans)):
            ans_cnt[list_ans[pos]] += 1
            mine_cnt[list_mine[pos]] += 1
            if list_ans[pos] == list_mine[pos]:
                true_cnt[list_ans[pos]] += 1

        for item in keys:
            if ans_cnt[item] != 0:
                num += 1
                if mine_cnt[item] == 0:
                    pi = 0
                else:
                    pi = true_cnt[item] / mine_cnt[item]
                ri = true_cnt[item] / ans_cnt[item]
                result[item] = 2 * pi * ri / (pi + ri)
                ans += 2 * pi * ri / (pi + ri)
        ans /= num
        result["总分"] = ans

        return result
