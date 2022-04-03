import os.path
import json
from sklearn.utils import shuffle

"""
from all.json to create training set and testing set
"""


def read_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(json.loads(line.strip()))  # strip()用于去掉两边多余空格
        return lines


def convert_tq_pair_to_rq_pair(datasets, out_file):
    """
    将table-query关系对转换为row-query关系对
    :param datasets: 原始数据集
    :param out_file: 处理后的数据集
    """
    for dataset in datasets:
        rel = dataset['rel']
        qid = dataset['qid']
        docid = dataset['docid']
        query = dataset['query']
        caption = dataset['table']['caption']
        title = json.loads(dataset['table']['raw_json'])['title']
        headings = []
        for heading in title:
            headings.append(heading)
        table_data = json.loads(dataset['table']['raw_json'])['data']
        for idx, row in enumerate(table_data):
            cells = []
            for cell in row:
                cells.append(cell)
            dic = {'rel': rel, 'qid': qid, 'docid': docid + ('-row_{0}'.format(idx)), 'query': query,
                   'row_data': cells, 'caption': caption, 'headings': headings}
            json.dump(dic, out_file)
            out_file.write('\n')


def create_ith_set(i, k, lines):
    ith_train_lines = []
    ith_test_lines = []
    for idx, line in enumerate(lines):
        # print("hello\n")
        if idx % k == i:
            ith_test_lines.append(line)
        else:
            ith_train_lines.append(line)
    return ith_train_lines, ith_test_lines


def divide_k_fold_sets(datasets_file, k=5):
    """
    划分k折交叉验证的集合
    :param datasets_file: 所有数据集
    :param k: k值，默认为5
    """
    # fp = open(datasets_file, "r")
    lines = []
    for line in datasets_file:
        lines.append(line)
    lines = shuffle(lines)
    for i in range(k):
        train_lines, test_lines = create_ith_set(i, k, lines)
        output_train_file = open(
            "./data/{0}_train.jsonl".format(i), "w")
        output_test_file = open(
            "./data/{0}_test.jsonl".format(i), "w")
        for line in train_lines:
            output_train_file.write(line)
        for line in test_lines:
            output_test_file.write(line)
        output_train_file.close()
        output_test_file.close()


def main():
    datasets = read_json("./data/all.json")
    processed_datasets_file = open(
        "./data/processed_all.jsonl", "w")
    convert_tq_pair_to_rq_pair(datasets, processed_datasets_file)
    processed_datasets_file.close()
    processed_datasets_file = open(
        "./data/processed_all.jsonl", "r")
    divide_k_fold_sets(processed_datasets_file)
    processed_datasets_file.close()


if __name__ == '__main__':
    main()
