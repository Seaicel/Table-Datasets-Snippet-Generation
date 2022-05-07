"""
用于构建wdc的训练集
"""


import json
from sklearn.utils import shuffle


def read_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(json.loads(line.strip()))  # strip()用于去掉两边多余空格
        return lines


def create_tid_table_dict(file_path):
    tid_table_dict = {}
    tid_header_dict = {}
    tid_orientation_dict = {}
    datasets = read_json(file_path)
    for dataset in datasets:
        tid = dataset['tid']
        table = json.loads(dataset['raw_json'])['relation']
        has_header = json.loads(dataset['raw_json'])['hasHeader']
        orientation = dataset['orientation']
        tid_table_dict[tid] = table
        tid_header_dict[tid] = has_header
        tid_orientation_dict[tid] = orientation
    return tid_table_dict, tid_header_dict, tid_orientation_dict


def create_queries_list(file_path):
    queries_list = ["beginning"]
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split()[1:]
            query = ' '.join(line)
            queries_list.append(query)
    return queries_list


# def link_query_table(tid_table_dict, queries_list, input_file, output_file):
#     fw = open(output_file, "w")
#     with open(input_file, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.split()
#             rel = float(line[3])
#             if rel != float(2.0):
#                 continue
#             query = queries_list[int(line[0])]
#             table = tid_table_dict[line[2]]
#             for idx, row in enumerate(table):
#                 # 处理空行
#                 is_empty = True
#                 for cell in row:
#                     if len(cell) != 0:
#                         is_empty = False
#                         break
#                 if is_empty:
#                     continue
#                 dic = {'rel': rel, 'qid': line[0], 'docid': line[2] + ('-row_{0}'.format(idx)),
#                        'query': query, 'row_data': row}
#                 json.dump(dic, fw)
#                 fw.write('\n')
#     fw.close()


def link_query_table(tid_table_dict, tid_header_dict, tid_orientation_dict, queries_list, input_file, output_file):
    fw = open(output_file, "w")
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split()
            rel = float(line[3])
            if rel != float(2.0):
                continue
            query = queries_list[int(line[0])]
            table = tid_table_dict[line[2]]
            row_dic_list = []
            if not tid_header_dict[line[2]]:  # 没有表头
                if tid_orientation_dict[line[2]] == "HORIZONTAL":
                    for col_idx, col in enumerate(table):
                        header = "NULL{0}".format(col_idx + 1)
                        for row_idx, row_cell in enumerate(col):
                            if col_idx == 0:
                                row_dic = {header: row_cell}
                                row_dic_list.append(row_dic)
                            else:
                                row_dic_list[row_idx][header] = row_cell

                elif tid_orientation_dict[line[2]] == "VERTICAL":
                    for row_idx, row in enumerate(table):
                        row_dic = {}
                        for col_idx, col_cell in enumerate(row):
                            header = "NULL{0}".format(col_idx + 1)
                            row_dic[header] = col_cell
                        row_dic_list.append(row_dic)

            else:  # 有表头
                if tid_orientation_dict[line[2]] == "HORIZONTAL":
                    for col_idx, col in enumerate(table):
                        header = ""
                        for row_idx, row_cell in enumerate(col):
                            if row_idx == 0:
                                header = row_cell
                            else:
                                if len(row_dic_list) < row_idx:
                                    row_dic = {header: row_cell}
                                    row_dic_list.append(row_dic)
                                else:
                                    row_dic_list[row_idx - 1][header] = row_cell

                elif tid_orientation_dict[line[2]] == "VERTICAL":
                    header = []
                    for row_idx, row in enumerate(table):
                        row_dic = {}
                        for col_idx, col_cell in enumerate(row):
                            if row_idx == 0:
                                header.append(col_cell)
                            else:
                                row_dic[header[col_idx]] = col_cell
                        row_dic_list.append(row_dic)

            dic = {'rel': rel, 'qid': line[0], 'docid': line[2],
                   'query': query, 'data': row_dic_list}
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
            "../data/wdc_{0}_train.jsonl".format(i), "w")
        output_test_file = open(
            "../data/wdc_{0}_test.jsonl".format(i), "w")
        for line in train_lines:
            output_train_file.write(line)
        for line in test_lines:
            output_test_file.write(line)
        output_train_file.close()
        output_test_file.close()
    datasets_file.close()


def main():
    tid_table_dict, tid_header_dict, tid_orientation_dict = create_tid_table_dict("../data/wdc_pool.json")
    queries_list = create_queries_list("../data/queries.txt")
    input_file = "../data/rel_table_qrels.txt"
    output_file = "../data/wdc_all_highly_rekevant.jsonl"
    link_query_table(tid_table_dict, tid_header_dict, tid_orientation_dict, queries_list, input_file, output_file)
    # divide_k_fold_sets("../data/wdc_all.jsonl")


if __name__ == '__main__':
    main()
