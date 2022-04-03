import os
import argparse
import datetime
import pandas as pd
import torch
import json
import re
import sys
# import tensorflow as tf
import logging
import shutil
from tensorboardX import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from pytorch_transformers import (WEIGHTS_NAME,
                                  BertConfig,
                                  BertTokenizer,
                                  BertForSequenceClassification,  # BERT Model
                                  )
from pytorch_transformers import AdamW, WarmupLinearSchedule
from tqdm import tqdm, trange
from collections import defaultdict, Counter
import random
import numpy as np

sys.path.append('../')
from trec import TREC_evaluator

"""
用于bert模型的训练
用TensorDataset来打包特征数据
参考 https://github.com/Brokenwind/BertSimilarity
"""

# 需要加载训练数据，将数据转换为特征，打包之后用DataLoader加载
# 用TensorDataset将数据打包，用DataLoader来加载


logger = logging.getLogger(__name__)


def read_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(json.loads(line.strip()))  # strip()用于去掉两边多余空格
        return lines


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, query, qid, docid, caption=None, headings=None, row_data=None, label=None):
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
        self.caption = caption
        self.headings = headings
        self.row_data = row_data
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def get_train_example(data_dir):
    datasets = read_json(data_dir)
    examples = []
    for idx, dataset in enumerate(datasets):
        guid = 'train-%d' % idx
        examples.append(InputExample(guid=guid, query=dataset['query'], qid=dataset['qid'],
                                     docid=dataset['docid'], caption=dataset['caption'],
                                     headings=dataset['headings'],
                                     row_data=dataset['row_data'],
                                     label=dataset['rel']))
    return examples


def get_test_examples(data_dir):
    datasets = read_json(data_dir)
    examples = []
    for idx, dataset in enumerate(datasets):
        guid = 'test-%d' % idx
        examples.append(InputExample(guid=guid, query=dataset['query'], qid=dataset['qid'],
                                     docid=dataset['docid'], caption=dataset['caption'],
                                     headings=dataset['headings'],
                                     row_data=dataset['row_data'],
                                     label=dataset['rel']))
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(args, examples, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""
    features = []
    for (ex_index, example) in enumerate(examples):
        # label_map = {}
        # for (i, label) in enumerate(label_list):
        #     label_map[label] = i

        tokens = ['CLS']

        pattern = re.compile('[\W_]+')
        row_data = ' '.join(example.row_data)
        row_data = pattern.sub(' ', row_data.lower())
        token_row_data = tokenizer.tokenize(row_data)

        # add query tokens
        token_query = tokenizer.tokenize(example.query.lower())
        _truncate_seq_pair(token_query, token_row_data,
                           max_seq_length - len(token_query) - 2)
        tokens += (token_query + ['SEP'])
        segment_ids = [0] * (len(token_query) + 2)

        # add caption tokens
        if args.add_caption and example.caption:
            token_caption = tokenizer.tokenize(example.caption.lower())
            tokens += token_caption + ['SEP']
            segment_ids += [1] * (len(token_caption) + 1)

        # add headings tokens
        if args.add_headings and example.headings:
            heading_row = ' '.join(example.headings)
            token_headings = tokenizer.tokenize(heading_row.lower())
            _truncate_seq_pair(token_headings, token_row_data,
                               max_seq_length - len(token_headings) - 2)
            tokens += token_headings + ['SEP']
            segment_ids += [1] * (len(token_headings) + 1)

        # add row data tokens
        tokens += token_row_data + ['SEP']
        segment_ids += [1] * (len(token_row_data) + 1)

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

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

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


def train(args, train_dataset, model, tokenizer):
    tb_fname = os.path.join('./runs', args.exp_name)
    if os.path.exists(tb_fname):
        shutil.rmtree(tb_fname)
    tb_writer = SummaryWriter(logdir=tb_fname)
    args.train_batch_size = 20
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    """
        learning rate layer decay, hard coding for BERT, XLNet and RoBERTa.
    """

    def extract_n_layer(n, max_n_layer=-1):
        n = n.split('.')
        try:
            idx = n.index("layer")
            n_layer = int(n[idx + 1]) + 1
        except:
            if any(nd in n for nd in ["embeddings", "word_embedding", "mask_emb"]):
                n_layer = 0
            else:
                n_layer = max_n_layer
        return n_layer
    # we acquire the max_n_layer from inference,
    # we leave the sequence_summary layer and logits layer own same learning rate scale 1.
    # the lower 24 encoder layers shave decaying learning rate scare decay_scale ** (24-layer), layer ~ (0,23)
    max_n_layer = max([extract_n_layer(n) for n, p in model.named_parameters()]) + 1
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = []  # group params by layers and weight_decay params.
    for n_layer in range(max_n_layer + 1):
        #### n_layer and decay
        optimizer_grouped_parameters.append({
            'params': [p for n, p in model.named_parameters() if (
                    extract_n_layer(n, max_n_layer=max_n_layer) == n_layer and not any(
                nd in n for nd in no_decay))],
            'weight_decay': args.weight_decay,
            'lr_decay': args.lr_layer_decay ** (max_n_layer - n_layer)
        })
        #### n_layer and no_decay
        optimizer_grouped_parameters.append({
            'params': [p for n, p in model.named_parameters() if (
                    extract_n_layer(n, max_n_layer=max_n_layer) == n_layer and any(nd in n for nd in no_decay))],
            'weight_decay': 0.0,
            'lr_decay': args.lr_layer_decay ** (max_n_layer - n_layer)
        })
        # #### debug info
        # ns = [n for n, _ in model.named_parameters() if (
        #     extract_n_layer(n, max_n_layer=max_n_layer) == n_layer and not any(nd in n for nd in no_decay))]
        # lr_decay = args.lr_layer_decay ** (max_n_layer-n_layer)
        # print(ns)
        # print(lr_decay)
        # print('\n\n')
    ## setting optimizer, plan to add RADAM & LookAhead
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    if 0. < args.warmup_proportion < 1.0:
        warmup_steps = t_total * args.warmup_proportion
    else:
        warmup_steps = args.warmup_steps
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    # logger.info("  Fold  = %d", args.fold)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d, warmup steps = %d", t_total, warmup_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    epoch_num = 0

    best_loss = float("inf")

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            # (loss), logits, (hidden_states), (attentions)
            # first_token_tensor = hidden_states[:, 0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer,
                                                prefix='test ' + str(global_step) + ' :epoch ' + str(epoch_num))
                        for key, value in results.items():
                            tb_writer.add_scalar('test_eval_{}'.format(key), value, global_step)
                        # train_results = evaluate(args, model, tokenizer, prefix='train: ' + str(global_step) + ' :epoch ' + str(epoch_num),split='train')
                        # for key, value in train_results.items():
                        #     tb_writer.add_scalar('train_eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                scheduler.step()

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        epoch_num += 1
        logger.info("epoch %s", epoch_num)

        results = evaluate(args, model, tokenizer,
                                prefix='epoch ' + str(epoch_num),
                                split='test')
        for key, value in results.items():
            tb_writer.add_scalar('test_eval_epoch_{}'.format(key), value, epoch_num)
        eval_loss = results["eval_loss"]
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, '{0}_best_model.pkl'.format(args.exp_name)))
        # train_results = evaluate(args, model, tokenizer, prefix='epoch ' + str(epoch_num), split='train')
        # for key, value in train_results.items():
        #     tb_writer.add_scalar('train_eval_epoch_{}'.format(key), value, epoch_num)
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", split="test"):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    eval_dataset, eval_example = convert_examples_to_features(args,
                                                              get_test_examples(args.test_dir)
                                                              , 128, tokenizer)
    args.eval_batch_size = 20
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.train_batch_size)

    logger.info("***** Running evaluation {} *****".format(prefix + '\t' + split))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

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
    qids = [eg.qid for eg in eval_example]
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

    return ltr_metric_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--add_caption", action='store_true', default=False, help="add caption to tokens.")
    parser.add_argument("--add_headings", action='store_true', default=False, help="add headings to tokens.")
    parser.add_argument("--data_dir", default=None, type=str, required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    #### learning rate difference between original BertAdam and now paramters.
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--lr_layer_decay", default=1.0, type=float,
                        help="layer learning rate decay.")

    args = parser.parse_args()
    args.model_name_or_path = './pre_trained/bert-large-cased'
    args.output_dir = './output'
    args.exp_name = 'Bert-Large-Cased_0_fold'
    # args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=1, finetuning_task='table')
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)

    # logger.setLevel(level = logging.INFO)
    # handler = logging.FileHandler("tmp.log")
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filemode='w')
    # add file handler to log training info
    if not os.path.isdir(args.output_dir): os.mkdir(args.output_dir)
    dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(os.path.join(args.output_dir, "{0}_training.log".format(dt)))
    logger.addHandler(fh)
    logger.warning("Device: %s",
                   args.device)

    # Set seed
    set_seed(args)

    args.train_dir = './data/00_train.jsonl'
    args.test_dir = './data/00_test.jsonl'
    examples = get_train_example(args.train_dir)
    train_dataset, _ = convert_examples_to_features(args, examples, 128, tokenizer)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # model_to_save = model.module if hasattr(model,
    #                                         'module') else model  # Take care of distributed/parallel training
    # model_to_save.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # evaluate
    results = evaluate(args, model, tokenizer, prefix='eval', split='00_test')
    # torch.save(results, os.path.join(args.output_dir, 'cache_rs_fold_0.bin'))
    # torch.save(model, )

if __name__ == '__main__':
    main()
