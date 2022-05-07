# from bert_cell_qrel_sort import evaluate
import re
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import torch


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, query, qid, docid, details=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
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


def get_test_examples(data_dir):
    datasets = read_json(data_dir)
    examples = []
    for idx, dataset in enumerate(datasets):
        guid = 'test-%d' % idx
        examples.append(InputExample(guid=guid, query=dataset['query'], qid=dataset['qid'],
                                     docid=dataset['docid'],
                                     details=dataset['details'],
                                     label=dataset['rel']))
    return examples


def _truncate_seq_pair(args, tokens_a, tokens_b, max_length):
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
            args.truncate_count += 1
            flag = False
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(args, examples, tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""
    features = []
    max_seq_length = args.max_seq_length
    for (ex_index, example) in enumerate(examples):
        # label_map = {}
        # for (i, label) in enumerate(label_list):
        #     label_map[label] = i

        tokens = ['CLS']

        pattern = re.compile('[\W_]+')
        details = ' '.join(example.details)
        details = pattern.sub(' ', details.lower())
        token_details = tokenizer.tokenize(details)

        # add query tokens
        token_query = tokenizer.tokenize(example.query.lower())

        '''
            TODO: 在截断之前根据相关性排序一下token_row_data
        '''
        # token_row_data = sort_tokens(args, token_query, token_row_data)
        # if args.tokens_sort_method == "RAND":
        #     random.shuffle(token_query)

        _truncate_seq_pair(args, token_query, token_details,
                           max_seq_length - len(tokens) - 2)
        tokens += (token_query + ['SEP'])
        segment_ids = [0] * (len(token_query) + 2)

        # add row data tokens
        tokens += token_details + ['SEP']
        segment_ids += [1] * (len(token_details) + 1)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
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
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
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


def evaluate():
    eval_dataset, eval_example = convert_examples_to_features(get_test_examples(args.test_dir)
                                                              , tokenizer)
    eval_batch_size = 32
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.train_batch_size)

    # logger.info("***** Running evaluation {} *****".format(prefix + '\t' + split))
    # logger.info("  Num examples = %d", len(eval_dataset))
    # logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0
    nb_eval_steps = 0
    scores = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      # XLM and RoBERTa don't use segment_ids
                      'labels': batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if scores is None:
            scores = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            scores = np.append(scores, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    # softmax score, for further processing (ensemble)

    # get scores or results depending on the task
    qids = [(eg.qid + '_' + eg.docid.split("_")[0]) for eg in eval_example]
    docids = [eg.docid for eg in eval_example]
    preds = scores.squeeze()
    # print(preds)
    # print(preds.shape)
    eval_df = pd.DataFrame(data={
        'id_left': qids,
        'id_right': docids,
        'true': out_label_ids,
        'pred': preds
    })
    ltr_metric_scores = defaultdict(list)

    #### Run evaluation
    trec_eval = TREC_evaluator(run_id=args.exp_name + '_' + split, base_path=args.output_dir)
    trec_eval.write_trec_result(eval_df)
    ndcgs = trec_eval.get_ndcgs()
    for metric in ndcgs:
        ltr_metric_scores[metric].append(ndcgs[metric])
    #
    ltr_metric_scores["eval_loss"] = eval_loss / nb_eval_steps
    # report resutls
    logger.info("***** Eval results {} *****".format(prefix + ' ' + split))
    for key in sorted(ltr_metric_scores.keys()):
        logger.info("  %s = %s", key, str(ltr_metric_scores[key]))


def main():
    evaluate()


if __name__ == '__main__':
    main()