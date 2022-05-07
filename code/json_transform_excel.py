import json
import pandas as pd
import os
from tqdm import tqdm

def read_jsonl(jsonl_file):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(json.loads(line.strip()))  # strip()用于去掉两边多余空格
        return lines


def json_out(file_path):
    """
    将json格式转换为xlsx格式
    :param path:
    :return:
    """
    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)
        print(data)
    data = pd.DataFrame(data)
    data.to_excel("../data/all_highly_relevant.xslx", index=None)


def json_outs(file_path):
    """
    将json格式转换为xlsx格式
    :param path:
    :return:
    """
    list_data = []
    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)
        data = pd.DataFrame(data)
        list_data.append(data)
    total_data = pd.concat(list_data)
    total_data.to_excel("../data/all_highly_relevant.xslx", index=None)


def creatExcelSheet(excelDataFilePath, data, sheet_name):
    """
        # excelDataFilePath: 原始表格文件
        # data: 自定义的数据，只要满足DataFrame格式的要求即可
        # sheet_name 需要创建的子表名称
    """
    # df = pd.DataFrame(data=data)
    # writer = pd.ExcelWriter(excelDataFilePath, mode='a', engine="openpyxl")
    # df.to_excel(writer, sheet_name=sheet_name)
    # 将数据写入原有表格的sheet_name子表中
    # df = pd.DataFrame(data={'a': [4], 'b': ['玉米'], 'c': [0.5]})
    df = pd.DataFrame(data=data)

    with pd.ExcelWriter(excelDataFilePath, mode='a', engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name)


if __name__ == '__main__':
    file_path = "../data/wdc_all_highly_rekevant.jsonl"
    json_datas = read_jsonl(file_path)
    query_num = [1] * 61
    json_iterator = tqdm(json_datas, desc="Iteration")
    for idx, json_data in enumerate(json_iterator):
        # print(idx)
        sheet_name = json_data['query'] + '_' + str(query_num[int(json_data['qid'])]) + '_' + str(idx)
        query_num[int(json_data['qid'])] += 1
        data = json_data['data']
        creatExcelSheet("../data/wdc_all_highly_relevant.xlsx", data, sheet_name)