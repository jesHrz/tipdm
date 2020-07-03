def mathch_sim(word_vector):
    import numpy as np
    import xlsxwriter

    vectors = word_vector
    # vectors = np.load('data/T3_full_vector.npy',allow_pickle=True)

    def cos_sim(x, y, norm=True):
        """ 计算两个向量x和y的余弦相似度 """
        assert len(x) == len(y), "len(x) != len(y)"
        # zero_list = [0] * len(x)
        if np.where(x != 0)[0].shape[0] == 0 or np.where(y != 0)[0].shape[0] == 0:
            return float(1) if all(x == y) else float(0)

        res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
        return 0.5 * cos + 0.5 if norm else cos

    res = []

    for i in range(len(vectors)):
        asks = vectors[i][0]
        anss = vectors[i][1]
        sum1 = 0
        sum2 = 0
        for ask in asks:
            max_sim = 0
            for ans in anss:
                max_sim = max(max_sim, cos_sim(ask, ans))
            sum1 += max_sim
        if len(asks) != 0:
            sum1 /= len(asks)

        for ans in anss:
            max_sim = 0
            for ask in asks:
                max_sim = max(max_sim, cos_sim(ask, ans))
            sum2 += max_sim
        if len(anss) != 0:
            sum2 /= len(anss)

        res.append((sum1 + sum2)/2)


    res = np.array(res)
    Zmax,Zmin=res.max(axis=0),res.min(axis=0)
    print(Zmin,Zmax)
    Z=(res-Zmin)/(Zmax-Zmin)

    return Z
    # workbook3 = xlsxwriter.Workbook('data/match_sim.xlsx')  # 创建一个excel文件
    # worksheet3 = workbook3.add_worksheet(u'sheet1')  # 在文件中创建一个名为TEST的sheet,不加名字默认为sheet1

    # for i in range((len(Z))):
    #     print(Z[i])
    #     worksheet3.write(i+1, 0, Z[i])

    # workbook3.close()