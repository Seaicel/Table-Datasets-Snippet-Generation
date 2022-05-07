import os.path
from collections import defaultdict

import fasttext
import json
import numpy as np
import pandas as pd
import scipy.spatial.distance as dis
import re
import argparse
from tfidf_keyword import TfIdf
from trec import TREC_evaluator

'''
Use cosine similarity to get the relevance 
ranking of cells/rows/cols and query.
Mean/Sum/Max

输入的是一些JSON表格数据和query，
输出是排序后的表格
'''

fasttext_model = fasttext.FastText.load_model("../pre_trained/wiki.simple/wiki.simple.bin")


def read_json(jsonl_file):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(json.loads(line.strip()))  # strip()用于去掉两边多余空格
        return lines


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, query, qid, docid, details=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.query = query
        self.qid = qid
        self.docid = docid
        self.details = details
        self.label = label


def sort_rows(args, table, model, fp):
    """
    ROW-MEAN 行向的平均方法
    :param table: 当前需要生成片段的表格
    :param model: 训练好的fattext模型
    最终向输出文件写入一个行相关性降序排列的表格
    """
    query = table['query'].lower().split()
    table_data = json.loads(table['table']['raw_json'])['data']
    pattern = re.compile('[\W_]+')
    idx_cosine = []
    # \W用来匹配非单词字符，等价于[^a-zA-Z0-9_]，所以'[\W_]+'为非数字和非字母
    for idx, row in enumerate(table_data):
        # mean
        cell_vec = []
        for cell in row:
            cell = pattern.sub(' ', cell)
            cell = cell.lower().split()
            this_cell_vec = np.mean([model.get_word_vector(token) for token in cell], axis=0)  # 按列求平均
            cell_vec.append(this_cell_vec)
        row_vec = np.mean(cell_vec, axis=0)
        query_vec = np.mean([model.get_word_vector(token) for token in query], axis=0)
        idx_cosine.append(dis.cosine(row_vec, query_vec))  # 越接近0，表示两个向量越相似
    idx_cosine = np.array(idx_cosine)
    indexs = np.argsort(idx_cosine)  # 按照与query的相似度对rows进行排序
    fp.write("Query: " + table['query'].lower() + "\n")
    for idx in indexs:
        row_data = ""
        for cell in table_data[idx]:
            row_data = (row_data + cell + "\t")
        row_data = (row_data[:-2] + "\n")
        fp.write(row_data)


def tfidf_keyword_sort_rows(args, table, model, fp):
    """
    用TF-IDF方法寻找每一行的关键tokens，与query的关键token进行余弦相似度的计算
    :param table: 当前需要生成片段的表格
    :param model: 训练好的fattext模型
    最终向输出文件写入一个行相关性降序排列的表格
    """
    my_tfidf = TfIdf(stopword_filename='data/stopwords.txt')
    pattern = re.compile('[\W_]+')

    # 将query以及table的每一行加入input
    query = table['query'].lower()
    my_tfidf.add_input_document(query)
    table_data = json.loads(table['table']['raw_json'])['data']
    rows_data = []
    for idx, row in enumerate(table_data):
        row_data = ""
        for cell in row:
            cell = pattern.sub(' ', cell.lower())
            row_data += (cell + ' ')
        rows_data.append(row_data)
        my_tfidf.add_input_document(row_data)

    # 得到query的关键字
    query_keyword = my_tfidf.get_doc_keywords(query)[0][0]
    query_vec = model.get_word_vector(query_keyword)

    # 得到每一行的top3关键字，计算平均vector，并与query关键字的vector进行余弦相似度计算
    idx_cosine = []
    for row_data in rows_data:
        row_keywords = my_tfidf.get_doc_keywords(row_data)
        row_vecs = []
        for i in range(3):
            if len(row_keywords) > i:
                row_vecs.append(model.get_word_vector(row_keywords[i][0]))
        row_vec = np.mean(row_vecs, axis=0)
        idx_cosine.append(dis.cosine(row_vec, query_vec))  # 越接近0，表示两个向量越相似
    idx_cosine = np.array(idx_cosine)
    indexs = np.argsort(idx_cosine)  # 按照与query的相似度对rows进行排序
    fp.write("Query: " + table['query'].lower() + "\n")
    for idx in indexs:
        row_data = ""
        for cell in table_data[idx]:
            row_data = (row_data + cell + "\t")
        row_data = (row_data[:-2] + "\n")
        fp.write(row_data)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_data", default="../data/0_test.jsonl",
#                         type=str, required=False, help="input data path.")
#     parser.add_argument("--fasttext_model", default="../pre_trained/wiki.simple/wiki.simple.bin",
#                         type=str, required=False, help=".bin fasttext model path.")
#     parser.add_argument("--output_path", default="../output",
#                         type=str, required=False, help="Store output data in this directory.")
#     parser.add_argument("--method", default=None, type=str, required=True,
#                         help="Method to calculate cosine similarity. ROW-MEAN or TF-IDF.")
#     # TF-IDF args
#     parser.add_argument("--topk", default=3, type=int, required=False,
#                         help="Get top k keywords to calculate row vector.")
#     args = parser.parse_args()
#
#     input_datas = read_json(args.input_data)
#     fp = open(os.path.join(args.output_path, 'cos_similarity_method_output.txt'), "w")
#     if args.method == "ROW-MEAN":
#         fp.write("Done ROW-MEAN method.\n")
#         for test in input_datas:
#             sort_rows(args, test, fasttext_model, fp)
#     elif args.method == "TF-IDF":
#         fp.write("Done TD-IDF method.\n")
#         for test in input_datas:
#             tfidf_keyword_sort_rows(args, test, fasttext_model, fp)
#     fp.close()


def cal_cos_sim_qcel(query, cell, method):
    query = query.lower().split()
    pattern = re.compile('[\W_]+')
    cell = pattern.sub(' ', cell.lower())
    cell = cell.split()
    # mean
    if method == "mean":
        cell_vec = np.mean([fasttext_model.get_word_vector(token) for token in cell], axis=0)
        query_vec = np.mean([fasttext_model.get_word_vector(token) for token in query], axis=0)
        return 1 - dis.cosine(cell_vec, query_vec)

    # sum
    elif method == "sum":
        cell_vec = np.mean([fasttext_model.get_word_vector(token) for token in cell], axis=0)
        cos_sim_sum = 0
        for query_token in query:
            query_token_vec = fasttext_model.get_word_vector(query_token)
            cos_sim_sum += (1 - dis.cosine(cell_vec, query_token_vec))
        return cos_sim_sum

    # max
    elif method == "max":
        max_cos_sim = 0
        cell_vec = np.mean([fasttext_model.get_word_vector(token) for token in cell], axis=0)
        for query_token in query:
            query_token_vec = fasttext_model.get_word_vector(query_token)
            # for cell_token in cell:
            #     cell_token_vec = fasttext_model.get_word_vector(cell_token)
            #     max_cos_sim = max(max_cos_sim, (1 - dis.cosine(cell_token_vec, query_token_vec)))
            max_cos_sim = max(max_cos_sim, 1 - dis.cosine(cell_vec, query_token_vec))
        return max_cos_sim
    #
    # elif method == "tfidf":
    #

    else:
        return 0


def get_cos_sim_ndcgs(input_path, method="mean"):
    datasets = read_json(input_path)
    examples = []
    qid_set = []
    for dataset in datasets:
        examples.append(InputExample(query=dataset['query'], qid=dataset['qid'],
                                     docid=dataset['docid'],
                                     details=dataset['details'],
                                     label=dataset['rel']))

    qids = [(eg.qid + '_' + eg.docid.split("_")[0]) for eg in examples]  # for set 1
    # qids = [(eg.qid + '_' + eg.docid.split(".json")[0]) for eg in examples]  # for set 2
    docids = [eg.docid for eg in examples]
    out_label_ids = [eg.label for eg in examples]
    preds = []
    for qid in qids:
        if qid not in qid_set:
            qid_set.append(qid)
    print(len(qid_set))
    for example in examples:
        cos_sim = cal_cos_sim_qcel(example.query, example.details, method)
        preds.append(cos_sim)

    eval_df = pd.DataFrame(data={
        'id_left': qids,
        'id_right': docids,
        'true': out_label_ids,
        'pred': preds
    })
    ltr_metric_scores = defaultdict(list)

    #### Run evaluation
    trec_eval = TREC_evaluator(run_id="get_cos_sim_{0}_ndcgs".format(method), base_path="../output")
    trec_eval.write_trec_result(eval_df)
    ndcgs = trec_eval.get_ndcgs()
    for metric in ndcgs:
        ltr_metric_scores[metric].append(ndcgs[metric])
    # report resutls
    print("***** {0} cos similarity NDCG results *****".format(method))
    for key in sorted(ltr_metric_scores.keys()):
        print("  {0} = {1}".format(key, str(ltr_metric_scores[key])))


def main():
    get_cos_sim_ndcgs("../tmp/wdc_cell_query_rel.jsonl", "max")


if __name__ == '__main__':
    main()
