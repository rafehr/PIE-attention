import os
import random
import torch
from torch import Tensor
from typing import Tuple, List, Dict
from os.path import join
from torch.nn.utils.rnn import pad_sequence

from utils import assert_equal_len

##################################################
# Defining types
##################################################

Data = List[Tuple[List[str], List[str], List[str],
                  List[str], List[str], str, str, str]]

Enc_Data = Tuple[Tuple[Tensor, Tensor, Tensor],
                 Tuple[List[str], List[str], List[str], str, int]]

Batch_Data = Tuple[Enc_Data, ...]

##################################################
# Reading data
##################################################


def read_data(data_path: str, split: str) -> Data:
    sents = load_feature(join(data_path, split, 'sentences.txt'))
    pie_idxs = load_feature(join(data_path, split, 'pie_idxs.txt'))
    pos = load_feature(join(data_path, split, 'pos_tags.txt'))
    deprels = load_feature(join(data_path, split, 'deprels.txt'))
    heads = load_feature(join(data_path, split, 'heads.txt'))
    labels = load_labels(join(data_path, split, 'labels.txt'))
    types = load_labels(join(data_path, split, 'pie_types.txt'))
    sent_ids = load_labels(join(data_path, split, 'sent_ids.txt'))
    assert_equal_len(sents, pos, deprels, heads, labels, types, sent_ids)
    data = [(t, i, p, d, h, l, ty, si) for t, i, p, d, h, l, ty, si in zip(
        sents, pie_idxs, pos, deprels, heads, labels, types, sent_ids)]
    return data


def read_data_raw(
        tok_path,
        labels_path,
        pos_path,
        dep_path,
        head_path,
        single_labels_path,
        types_path,
        sent_ids_path,
        split):
    tok = load_feature(tok_path, split)
    idxs = load_feature(labels_path, split)
    pos = load_feature(pos_path, split)
    dep = load_feature(dep_path, split)
    head = load_feature(head_path, split)
    lab = load_labels(single_labels_path, split)
    types = load_labels(types_path, split)
    sent_ids = load_labels(sent_ids_path, split)
    assert_equal_len(tok, pos, dep, head, lab, types, sent_ids)
    data = [
        (t, i, p, d, h, l, ty, si) for t, i, p, d, h, l, ty, si
        in zip(tok, idxs, pos, dep, head, lab, types, sent_ids)
    ]
    return data


def load_feature(feat_path, enc='utf-8') -> List[List[str]]:
    with open(feat_path, 'r', encoding=enc) as f:
        feature = [i.split() for i in f.read().strip().split('\n')]
    return feature


def load_labels(feat_path, enc='utf-8') -> List[str]:
    with open(feat_path, 'r', encoding=enc) as l:
        labels = l.read().strip().split('\n')
    return labels

##################################################
# Encoding
##################################################


def encode_with(
        data: Data,
        word_enc: Dict[str, int],
        label_enc: Dict[str, int]
) -> Enc_Data:
    enc_data = []
    for inp, idxs, pos, deps, heads, out, typ, sent_i in data:
        enc_inp = torch.tensor([word_enc[w] for w in inp])
        enc_idxs = torch.tensor([int(i) for i in idxs])
        enc_out = torch.tensor(label_enc[out])
        enc_data.append(((enc_inp, enc_idxs, enc_out),
                        (pos, deps, heads, typ, sent_i)))
    return enc_data

##################################################
# Batching
##################################################


def data_iterator(
        data: Enc_Data,
        batch_size: int,
        shuffle: bool = False
) -> Batch_Data:
    order = list(range(len(data)))
    if shuffle:
        random.seed(42)
        random.shuffle(order)
    else:
        data.sort(key=lambda x: len(x[0][0]))

    # One pass over data
    for b in range((len(data)//batch_size) + 1):
        sents = [data[i][0][0]
                 for i in order[b * batch_size: (b+1) * batch_size]]
        pie_idxs = [data[i][0][1]
                    for i in order[b * batch_size: (b+1) * batch_size]]
        labels = [data[i][0][2]
                  for i in order[b * batch_size: (b+1) * batch_size]]
        pos = [data[i][1][0]
               for i in order[b * batch_size: (b+1) * batch_size]]
        deprels = [data[i][1][1]
                   for i in order[b * batch_size: (b+1) * batch_size]]
        heads = [data[i][1][2]
                 for i in order[b * batch_size: (b+1) * batch_size]]
        types = [data[i][1][3]
                 for i in order[b * batch_size: (b+1) * batch_size]]
        sent_ids = [data[i][1][4]
                    for i in order[b * batch_size: (b+1) * batch_size]]

        sents = pad_sequence(sents, batch_first=True, padding_value=0.0)
        labels = torch.tensor(labels)
        yield sents, pie_idxs, labels, (pos, deprels, heads, types, sent_ids)
