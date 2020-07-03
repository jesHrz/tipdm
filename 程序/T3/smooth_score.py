import xlrd
import xlsxwriter
import numpy as np
from tqdm import tqdm
from myltp import MyLTP

def grammer_dict_create(words, postags, arcs):
    result = []
    for index in range(0, len(words)):
        if arcs[index].head == 0:
            A = -1
        else:
            A = postags[arcs[index].head-1]
        B = postags[index]
        C = arcs[index].relation
        result.append((A,B,C))
    return result

def semantic_dict_create(words, postags, arcs):
    result = []
    for index in range(0, len(words)):
        if arcs[index].head == 0:
            A = -1
        else:
            A = words[arcs[index].head-1]
        B = words[index]
        C = arcs[index].relation
        result.append((A,B,C))
    return result

def calc_grammer_value(data_set, grammar_path, Myltp):
    grammar_dic = np.load(grammar_path, allow_pickle=bool)["dic"][()]
    result = []
    for line in tqdm(data_set):
        words = Myltp.MySegmentor(line)
        postags = Myltp.MyPostagger(words)
        arcs = Myltp.MyParser(words, postags)
        totalpair, rightpair = 0, 0
        for index in range(0, len(words)):
            tp_list = grammer_dict_create(words[index], postags[index], arcs[index])
            for key in tp_list:
                totalpair += 1
                if grammar_dic.get(key, -1) != -1:
                    rightpair += 1
        if totalpair == 0:
            result.append(0)
        else:
            result.append(rightpair/totalpair)
    return result

def semantic_dict_create(words, postags, arcs):
    result = []
    for index in range(0, len(words)):
        if arcs[index].head == 0:
            A = -1
        else:
            A = words[arcs[index].head-1]
        B = words[index]
        C = arcs[index].relation
        result.append((A,B,C))
    return result

def calc_semantic_value(data_set, semantic_path, Myltp):
    semantic_dic = np.load(semantic_path, allow_pickle=bool)["dic"][()]
    result = []
    for line in tqdm(data_set):
        words = Myltp.MySegmentor(line)
        postags = Myltp.MyPostagger(words)
        arcs = Myltp.MyParser(words, postags)
        totalpair, rightpair = 0, 0
        for index in range(0, len(words)):
            tp_list = semantic_dict_create(words[index], postags[index], arcs[index])
            for key in tp_list:
                totalpair += 1
                if semantic_dic.get(key, -1) != -1:
                    rightpair += 1
        if totalpair == 0:
            result.append(0)
        else:
            result.append(rightpair/totalpair)
    return result

def data_acquire(path):
    result, data = [], []
    sheet = xlrd.open_workbook(path).sheet_by_index(0)
    for line in range(1, sheet.nrows):
        tmp = []
        for j in range(0, 7):
            tmp.append(sheet.cell(line, j).value)
        result.append(tmp)
        data.append(tmp[5])
    return data, result

def smooth_score():
    Myltp = MyLTP()
    grammar_path = "./data/corpus_grammar_small.npz" # 数据集太大，未上传
    semantic_path = "./data/corpus_semantic.npz" # 数据集太大，未上传

    data_load_path = "./data/附件4.xlsx"
    # data_save_path = "./data/corpus_result.xlsx"
    data_set, total_data = data_acquire(data_load_path)
    
    data_grammer_result = calc_grammer_value(data_set, grammar_path, Myltp)
    data_semantic_result = calc_semantic_value(data_set, semantic_path, Myltp)
    result = []

    for index in range(0, len(data_set)):
        result[index].append(data_grammer_result[index])
        result[index].append(data_semantic_result[index])
        result[index].append((data_grammer_result[index]+data_semantic_result[index])/2)
    
    return result