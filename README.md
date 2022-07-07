# Table-Datasets-Snippet-Generation
 
### 文件夹介绍（最终评测集在tmp文件夹）
#### 1. code 
包含了处理表格数据集内容、生成评测集、相似度计算、bert模型训练相关的代码。<br/>

#### 2. tmp
包含了诸多评测集以及其原始的xlsx表格数据。<br/><br/>
其中，名字中有"wdc"的评测集，原始的查询-表格相关性标记结果来自论文<a herf = "https://arxiv.org/abs/2105.02354">WTR: A Test Collection for Web Table Retrieval</a>.<br/>
名字中没有"wdc"的评测集，原始的查询-表格相关性标记结果来自论文<a herf = "https://arxiv.org/abs/1802.06159v3"> Ad Hoc Table Retrieval using Semantic Similarity</a>.<br/>
两个评测集的表格数据都来自<a herf = "https://dblp.uni-trier.de/rec/conf/semweb/BhagavatulaND15.html"> WikiTables</a>.<br/>
<br/>
"all_highly_relevant.xlsx" 和 "wdc_all_highly_relevant.xlsx":<br/>
包含了从原始查询-表格相关性标记结果中筛选出来的与查询相关度为2的表格，并且将每一个cell与查询的相关性进行了标记。<br/>
<br/>
"all_cell_rel.json" 和 "wdc_all_cell_rel.json"：<br/>
不重要，当作评测集生成的中间产物即可。记录了查询和表格中每个cell的相关度。<br/>
<br/>
"cell_query_rel.jsonl" 和 "wdc_cell_query_rel.jsonl":<br/>
最终生成的评测集。前者包含25859个查询-单元格相关度，后者包含99821个查询-单元格相关度。<br/>

#### 3. data
前缀为"cell"以及"wdc_cell"的文件是与查询-单元格相关度有关的json评测集文件，其余文件不必理会。<br/>
评测集做了 5-fold 分割。<br/>

