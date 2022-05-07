"""
用于生成cell-query相关度评测集
摈弃数字列除了表头之外的cell
训练集
"""

import json
import  re
from sklearn.utils import shuffle

uint_words = {'m', 'cm', 'mm', 'km', 'dm',
              'mg', 'g', 'kg', 't',
              'j', 'kj', 'cal', 'kcal',
              'ml', 'l',
              's', 'min', 'am', 'pm'
              'lb', 'in',
              'b', 'bit', 'byte', 'k', 'kb', 'mb', 'gb',
              'mpg', 'iu'}


def read_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(json.loads(line.strip()))  # strip()用于去掉两边多余空格
        return lines


def is_digit_col(table_data, col_num):
    data = table_data['data']
    pattern = re.compile('[^A-Za-z]+')
    non_digit_cell_num = 0  # 不为数字的cell数
    has_detials_cell_num = 0  # 不为空的cell数
    for row_data in data:
        cell_str = row_data[col_num]['details']
        if cell_str == 'None':  # None不算是
            continue
        has_detials_cell_num += 1
        cell_str = pattern.sub('', cell_str).lower()
        cell_str_list = cell_str.split()
        is_unit = True
        for s in cell_str_list:
            if s not in uint_words:
                is_unit = False
                break
        if len(cell_str) != 0 and is_unit is False:
            non_digit_cell_num += 1
    # 如果非数字cell占用整个不为空的cell数的0.2及以下，就表示是数字列
    if has_detials_cell_num * 0.2 >= non_digit_cell_num:
        return True
    return False


def process_cell_qrel(input_file, output_file):
    table_datas = read_json(input_file)
    fw = open(output_file, 'w')
    for table_idx, data in enumerate(table_datas):
        for i in range(data['num_cols']):
            if data['head'][i]['details'] is not None and data['head'][i]['details'] != "":
                rel = data['head'][i]['rel']
                details = data['head'][i]['details']
                qid = data['qid']
                query = data['query']
                docid = data['docid'] + '_{0}_head'.format(i)
                dic = {'rel': rel, 'qid': qid, 'query': query, 'docid': docid, 'details': details}
                json.dump(dic, fw)
                fw.write('\n')
            if not is_digit_col(data, i):
                # 该列是非数字列，都要加
                for idx, row_data in enumerate(data['data']):
                    if row_data[i]['details'] == 'None':  # 不加空cell
                        continue
                    rel = row_data[i]['rel']
                    details = row_data[i]['details']
                    qid = data['qid']
                    query = data['query']
                    docid = data['docid'] + '_{0}_{1}_{2}'.format(table_idx, i, idx)
                    dic = {'rel': rel, 'qid': qid, 'query': query, 'docid': docid, 'details': details}
                    json.dump(dic, fw)
                    fw.write('\n')
    fw.close()


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


def divide_k_fold_sets(input_file, k=5):
    """
    划分k折交叉验证的集合
    :param input_file: 所有数据集
    :param k: k值，默认为5
    """
    datasets_file = open(input_file, "r")
    lines = []
    for line in datasets_file:
        lines.append(line)
    lines = shuffle(lines)
    for i in range(k):
        train_lines, test_lines = create_ith_set(i, k, lines)
        output_train_file = open(
            "../data/wdc_cell_qrel_{0}_train.jsonl".format(i), "w")
        output_test_file = open(
            "../data/wdc_cell_qrel_{0}_test.jsonl".format(i), "w")
        for line in train_lines:
            output_train_file.write(line)
        for line in test_lines:
            output_test_file.write(line)
        output_train_file.close()
        output_test_file.close()
    datasets_file.close()


def main():
    input_file = "../tmp/wdc_all_cell_rel.json"
    output_file = "../tmp/wdc_cell_query_rel.jsonl"
    process_cell_qrel(input_file, output_file)
    divide_k_fold_sets(output_file, 5)


if __name__ == '__main__':
    main()