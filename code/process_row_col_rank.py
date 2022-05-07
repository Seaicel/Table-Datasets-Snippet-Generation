"""
表中每一个cell，都有标记一个相关度0-2
最后会添加一个列排序和行排序
"""
import json
import re
from create_cell_qrel_train_test import is_digit_col


uint_words = {'m', 'cm', 'mm', 'km', 'dm',
              'mg', 'g', 'kg', 't', 'mcg'
              'j', 'kj', 'cal', 'kcal',
              'ml', 'l',
              's', 'min', 'am', 'pm'
              'lb', 'in',
              'b', 'bit', 'byte', 'k', 'kb', 'mb', 'gb',
              'mpg', 'iu'}


def read_jsonl(jsonl_file):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(json.loads(line.strip()))  # strip()用于去掉两边多余空格
        return lines


# def is_digit_col(table_data, col_num):
#     data = table_data['data']
#     pattern = re.compile('[^A-Za-z]+')
#     non_digit_cell_num = 0
#     for row_data in data:
#         cell_str = row_data[col_num]['details']
#         if cell_str == 'None':
#             continue
#         cell_str = pattern.sub('', cell_str).lower()
#         cell_str_list = cell_str.split()
#         is_unit = True
#         for s in cell_str_list:
#             if s not in uint_words:
#                 is_unit = False
#                 break
#         if len(cell_str) != 0 and is_unit is False:
#             non_digit_cell_num += 1
#     # 如果非数字行占用整个num_rows的0.2及以下，就表示是数字行
#     if table_data['num_rows'] * 0.2 >= non_digit_cell_num:
#         return True
#     return False


def find_max_rel(rel_dic, col_num):
    max_rel = rel_dic['head_rel'][col_num]
    for row_rel in rel_dic['cell_rel']:
        max_rel = max(max_rel, row_rel[col_num])
    return max_rel


def cal_col_row_rel(table_data, rel_dic):
    """
    输入一张每一个cell都被标记过相关度对表，最终返回行列的相关度
    :param table_data:
           rel_dic: 预测出来的相关度
    :return:
        col_rel: 列相关度list
        row_rel: 行相关度list
    """

    col_rel = []
    row_rel = []

    # 先对列进行计算，判断数字列or非数字列
    # 非数字列：列相关度取该列相关度最大的cell的值
    # 数字列：取表头相关度
    # 非数字列的非空cell不会出现-inf的相关度
    # 直接摈弃掉-inf的
    for i in range(table_data['num_cols']):
        if is_digit_col(table_data, i):
            rel = rel_dic['head_rel'][i]
        else:
            rel = find_max_rel(rel_dic, i)
        col_rel.append(rel)

    # 再对行进行计算，
    # 将每行cell的相关度取平均排序
    # 可能会有-inf的相关度，表示所在的列是数字列或者此cell为空
    for row_rels in rel_dic['cell_rel']:
        rel = float(0)
        for cell_rel in row_rels:
            if cell_rel != float('-inf'):
                rel += float(cell_rel)
        row_rel.append(rel)

    return col_rel, row_rel

#
# def main():
#     table_datas = read_jsonl('/Users/dongshuhan/CODE/Python/Table-Datasets-Snippet-Generation/tmp/all_cell_rel.json')
#     for data in table_datas:
#         col_rel, row_rel = cal_col_row_rel(data)
#         print(col_rel)
#         print(row_rel)
#
#
# if __name__ == '__main__':
#     main()