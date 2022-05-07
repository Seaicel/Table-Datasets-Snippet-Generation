import json
import tablib
import pandas as pd

def read_jsonl(jsonl_file):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(json.loads(line.strip()))  # strip()用于去掉两边多余空格
        return lines


def process_all_highly_relevant(input_file, output_file):
    input_data = read_jsonl(input_file)
    fw = open(output_file, "w")
    for dataset in input_data:
        rel = dataset['rel']
        if rel != "2":
            continue
        qid = dataset['qid']
        docid = dataset['docid']
        query = dataset['query']
        caption = dataset['table']['caption']
        title = json.loads(dataset['table']['raw_json'])['title']
        table_data = json.loads(dataset['table']['raw_json'])['data']
        json_list = []
        for data in table_data:
            row_dic = {}
            for idx, cell in enumerate(data):
                header = title[idx]
                row_dic[header] = cell
            json_list.append(row_dic)
        dic = {'rel': rel, 'qid': qid, 'docid': docid, 'query': query,
               'caption': caption, 'data': json_list}
        json.dump(dic, fw)
        fw.write('\n')
    fw.close()


def wdc_process_all_highly_relevant(input_file, output_file):
    input_data = read_jsonl(input_file)
    fw = open(output_file, "w")
    for dataset in input_data:
        rel = dataset['rel']
        if rel != "2":
            continue
        qid = dataset['qid']
        docid = dataset['docid']
        query = dataset['query']
        caption = dataset['table']['caption']
        title = json.loads(dataset['table']['raw_json'])['title']
        table_data = json.loads(dataset['table']['raw_json'])['data']
        json_list = []
        for data in table_data:
            row_dic = {}
            for idx, cell in enumerate(data):
                header = title[idx]
                row_dic[header] = cell
            json_list.append(row_dic)
        dic = {'rel': rel, 'qid': qid, 'docid': docid, 'query': query,
               'caption': caption, 'data': json_list}
        json.dump(dic, fw)
        fw.write('\n')
    fw.close()


def main():
    input_file = "/Users/dongshuhan/CODE/Python/SIGIR2020-BERT-Table-Search/data/all.json"
    output_file = "../data/all_highly_relevant.json"
    process_all_highly_relevant(input_file, output_file)


if __name__ == '__main__':
    main()
