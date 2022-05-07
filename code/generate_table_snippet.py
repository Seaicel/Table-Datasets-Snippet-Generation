"""
使用训练完成的bert模型来预测每一个cell的相关度，
并调用process_row_col_rank中的方法对行列进行排序。
最终返回一个行列排序完成对表格，
可以指定片段行列数。
"""
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange

from trec import TREC_evaluator
from process_row_col_rank import is_digit_col, cal_col_row_rel
from pytorch_transformers import (BertConfig,
                                  BertTokenizer,
                                  BertForSequenceClassification,  # BERT Model
                                  )


def read_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def get_input_table_cells(table_data):
    examples = []
    for i in range(table_data['num_cols']):
        if table_data['head'][i]['details'] is not None:
            rel = table_data['head'][i]['rel']
            details = table_data['head'][i]['details']
            qid = table_data['qid']
            query = table_data['query']
            docid = table_data['docid'] + '_{0}_head'.format(i)
            examples.append(InputExample(query=query, qid=qid,
                                         docid=docid,
                                         details=details,
                                         label=rel))
        if not is_digit_col(table_data, i):
            # 该列是非数字列，都要加
            for idx, row_data in enumerate(table_data['data']):
                if row_data[i]['details'] == 'None':  # 不加空cell
                    continue
                rel = row_data[i]['rel']
                details = row_data[i]['details']
                qid = table_data['qid']
                query = table_data['query']
                docid = table_data['docid'] + '_{0}_{1}'.format(i, idx)
                examples.append(InputExample(query=query, qid=qid,
                                             docid=docid,
                                             details=details,
                                             label=rel))
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    flag = True
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if flag:
            flag = False
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_cell_features(examples, tokenizer):
    """Convert `InputExample`s to `InputFeatures`."""
    features = []
    max_seq_length = 128
    for (ex_index, example) in enumerate(examples):
        tokens = ['CLS']

        pattern = re.compile('[\W_]+')
        details = ' '.join(example.details)
        details = pattern.sub(' ', details.lower())
        token_details = tokenizer.tokenize(details)

        # add query tokens
        token_query = tokenizer.tokenize(example.query.lower())

        _truncate_seq_pair(token_query, token_details,
                           max_seq_length - len(tokens) - 2)
        tokens += (token_query + ['SEP'])
        segment_ids = [0] * (len(token_query) + 2)

        # add row data tokens
        tokens += token_details + ['SEP']
        segment_ids += [1] * (len(token_details) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # 要确保这三个一样长，都等于最大seq长度
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = float(example.label)
        # if ex_index < 1:
        #     logger.info("*** Input data: ***")
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id))

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset, examples


def write_trec_result(qid, eval_df, rank_path, qrel_path):
    # write result files
    f_rank = open(rank_path, 'w')
    f_rel = open(qrel_path, 'w')
    eval_df = eval_df.sort_values(by=['pred'], ascending=False)

    f_rank.write("qid" + "\t" + "rank" + "\t" + "docid" + "\t" + "pred" + "\t" + "true" + "\n")
    for idx, each in enumerate(eval_df.values):
        f_rank.write(each[0] + "\t" + str(idx + 1) + "\t" + each[1] + "\t" + str(each[3]) + "\t" + str(each[2]) + "\n")

    f_rank.close()


def predict_and_gen_ranked_table(input_path):
    input_tables = read_json(input_path)
    model_name_or_path = '../pre_trained/bert-large-cased'
    no_cuda = False
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")

    config = BertConfig.from_pretrained(model_name_or_path, num_labels=1, finetuning_task='table')
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    model = BertForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    model.load_state_dict(torch.load('../models/cell_0_fold_seq128_size32_ndcg10_08699.pkl', map_location=device))
    model.to(device)
    print("device: %s", device)

    eval_batch_size = 1

    all_qids = []
    all_docids = []
    all_preds = []
    all_labels = []
    fw = open("../output/col_row_rank.txt", "w")
    for table_idx, input_table in enumerate(input_tables):
        # 对每一个输入表做预测
        # 先将输入表中的每一个cell进行编码，预测相关度
        print("process table {0} ".format(table_idx + 1))
        eval_dataset, eval_example = convert_cell_features(get_input_table_cells(input_table), tokenizer)
        if len(eval_example) == 0:
            print("this table has no example cell.")
            fw.write("\n")
            continue
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        print("***** Running predict {} *****".format(input_table['docid']))
        print("  Num examples =", len(eval_dataset))
        print("  Batch size =", eval_batch_size)

        scores = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Predicting"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2]}
                outputs = model(**inputs)
                logits = outputs[0]

            if scores is None:
                scores = logits.detach().cpu().numpy()
                out_label_ids = batch[3].detach().cpu().numpy()
            else:
                scores = np.append(scores, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)

        qids = [(eg.qid + '_' + eg.docid.split("_")[0]) for eg in eval_example]
        docids = [eg.docid for eg in eval_example]
        if len(scores) != 1:
            preds = scores.squeeze()
        else:
            preds = scores[0]

        eval_df = pd.DataFrame(data={
            'id_left': qids,
            'id_right': docids,
            'true': out_label_ids,
            'pred': preds
        })

        for qid in qids:
            all_qids.append(qid)
        for docid in docids:
            all_docids.append(docid)
        for pred in preds:
            all_preds.append(pred)
        for ids in out_label_ids:
            all_labels.append(ids)

        col_rel, row_rel, col_idx, row_idx = sort_table(input_table, eval_df)

        for idx in col_idx:
            fw.write(str(idx))
            fw.write(" ")
        fw.write(",")
        for idx in row_idx:
            fw.write(str(idx))
            fw.write(" ")
        fw.write("\n")

        # write_res_table_to_excel(input_table, col_idx, row_idx)

    all_eval_df = pd.DataFrame(data={
        'id_left': all_qids,
        'id_right': all_docids,
        'true': all_labels,
        'pred': all_preds
    })
    ltr_metric_scores = defaultdict(list)

    trec_eval = TREC_evaluator(run_id="gen_table_snippet_ndcgs", base_path="../output")
    trec_eval.write_trec_result(all_eval_df)
    ndcgs = trec_eval.get_ndcgs()
    for metric in ndcgs:
        ltr_metric_scores[metric].append(ndcgs[metric])
    # report resutls
    print("***** table snippet NDCG results *****")
    for key in sorted(ltr_metric_scores.keys()):
        print("  {0} = {1}".format(key, str(ltr_metric_scores[key])))

    fw.close()


def sort_table(table, eval_df):
    num_cols = table['num_cols']
    num_rows = table['num_rows']

    head_rel = [float("-inf")] * num_cols
    cell_rel = [[float("-inf") for i in range(num_cols)] for i in range(num_rows)]

    for each in eval_df.values:
        col, row = each[1].split("_")[-2:]
        if row == "head":
            head_rel[int(col)] = float(each[3])
        else:
            cell_rel[int(row)][int(col)] = float(each[3])
    rel_dic = {'head_rel': head_rel, 'cell_rel': cell_rel}
    col_rel, row_rel = cal_col_row_rel(table, rel_dic)
    col_idx = (np.argsort(col_rel))[::-1]
    row_idx = (np.argsort(row_rel))[::-1]
    return col_rel, row_rel, col_idx, row_idx


def create_excel_sheet(excel_data_file_path, data, sheet_name):
    """
        # excelDataFilePath: 原始表格文件
        # data: 自定义的数据，只要满足DataFrame格式的要求即可
        # sheet_name 需要创建的子表名称
    """
    df = pd.DataFrame(data=data)

    with pd.ExcelWriter(excel_data_file_path, mode='a', engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name)


def write_res_table_to_excel(table_data, col_idx, row_idx):
    head = []
    for idx in col_idx:
        head.append(table_data['head'][idx]['details'])
    row_datas = []
    for idx in row_idx:
        dic = {}
        for col in col_idx:
            cell = table_data['data'][idx][col]['details']
            dic[table_data['head'][col]['details']] = cell
        row_datas.append(dic)
    sheet_name = table_data['query'] + '_' + "seq{0}".format(table_data['seq'])
    create_excel_sheet("../output/table_snippet.xlsx", row_datas, sheet_name)


def main():
    predict_and_gen_ranked_table("../tmp/all_cell_rel.json")
    # predict_and_gen_ranked_table("../tmp/test_table.json")


if __name__ == '__main__':
    main()