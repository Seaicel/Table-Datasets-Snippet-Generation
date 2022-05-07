# import re
#
# # pattern = re.compile('[\W_]+')
# pattern = re.compile('[^A-Za-z]+')
# # pattern = re.compile('[\W]+')
# str1 = '13243311_&*q'
# str1 = pattern.sub('', str1)
# print(bool(len(str1) == 0))
#
# import pandas as pd
# import dataframe_image as dfi
#
# qids = [1, 2, 3]
# docids = [11, 22, 33]
# out_label_ids = [0, 1, 2]
# preds = [0.1, 0.4, 1.1]
#
#
# eval_df = pd.DataFrame(data={
#     'id_left': qids,
#     'id_right': docids,
#     'true': out_label_ids,
#     'pred': preds
# })
#
# dfi.export(eval_df, "test.png")

count = 0
with open("../data/rel_table_qrels.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.split()
        rel = float(line[3])
        if rel == float(0.0):
            count += 1

print(count)