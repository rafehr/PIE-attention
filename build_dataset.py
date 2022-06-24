"""Reads the COLF-VID 1.0 corpus and creates a balanced split."""

import argparse
import random
import os
import json
from typing import Tuple, Set, List, Dict, Union
from os.path import join, basename
from collections import Counter
from utils import write_dict_to_json

PAD_WORD = '<pad>'
UNK_WORD = 'UNK'

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--corpus_dir', help='Directory of COLF-VID 1.0')
arg_parser.add_argument('--out_dir', default='data', help='Output directory')

##################################################
# Reading data
##################################################


def read_corpus(data_path: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """Reads the COLF files and returns a dictionary with lists of
    sentences (one list per file) as well as a list of PIE types.
    """
    file_paths = [join(data_path, f) for f in os.listdir(data_path)]
    sents_per_type = {basename(fp)[:-4]: read_file(fp) for fp in file_paths}
    pie_types = list(sents_per_type.keys())
    return sents_per_type, pie_types


def read_file(file_path: str) -> List[str]:
    """Reads a single COLF-VID 1.0 file and splits the sentences."""
    with open(file_path, 'r', encoding='utf-8') as f:
        assert file_path.endswith(".txt")
        sents = f.read().strip().split('\n\n')
    return sents

##################################################
# Split data and extract features
##################################################


def balanced_split(
        sents_per_type: Dict[str, List[str]],
        train_size: float = 0.7,
        test_size: float = 0.15,
        shuffle: bool = True,
        seed: int = 13
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    """Creates a balanced split of the dataset, i.e. the splits contain
    the same ratio of instances per type.
    """
    # Shuffle the sentences per type
    if shuffle:
        random.seed(seed)
        for pie_type in sents_per_type:
            random.shuffle(sents_per_type[pie_type])
    split_data = {'train': [], 'dev': [], 'test': []}
    num_instances_per_type = {}
    dev_size = 1. - train_size - test_size

    for pie_type, sents in sents_per_type.items():
        train_sents = sents[: int(train_size * len(sents))]
        dev_sents = sents[int(train_size * len(sents)):
                          int((train_size + dev_size) * len(sents))]
        test_sents = sents[int((train_size + dev_size) * len(sents)):]

        split_data['train'].extend(train_sents)
        split_data['dev'].extend(dev_sents)
        split_data['test'].extend(test_sents)

        num_instances_per_type[pie_type] = {
            "train": len(train_sents),
            "dev": len(dev_sents),
            "test": len(dev_sents)}
    return split_data, num_instances_per_type


def extract_feature(sent: str, col: int) -> List[str]:
    """Extract feature from a sentence in column format."""
    feature = [l.strip().split('\t')[col]
               for l in sent.split('\n') if not l.startswith('#')]
    return feature


def extract_features(
        split_data: Dict[str, List[str]]
) -> Tuple[Dict[str, Dict[str, List[str]]], Set[str]]:
    """Extracts the features of interest from the split dataset."""
    data = {'train': {}, 'dev': {}, 'test': {}}
    label_set = set()
    for split in split_data:
        sents, pos_tags, pie_idxs, labels, sent_ids, pie_types = (
            [] for _ in range(6))
        for sent in split_data[split]:
            sents.append(extract_feature(sent, 1))
            pos_tags.append(extract_feature(sent, 3))
            labels_per_token = extract_feature(sent, 7)
            for l in labels_per_token:
                label_set.add(l)
            pie_idxs.append(fetch_pie_idxs(labels_per_token))
            labels.append(fetch_single_label(labels_per_token))
            sent_ids.append(extract_from_comment(sent, '# sent_id = '))
            pie_types.append(extract_from_comment(sent, '# vid_types = '))
        data[split]['sentences'] = sents
        data[split]['pos_tags'] = pos_tags
        data[split]['pie_idxs'] = pie_idxs
        data[split]['labels'] = labels
        data[split]['sent_ids'] = sent_ids
        data[split]['pie_types'] = pie_types
        assert_len(data, split, 'sentences', 'pos_tags')
    return data, label_set


def extract_from_comment(sent: str, attr: str) -> Union[str, None]:
    """Extract the sentence ID from a sentence in column format."""
    for l in sent.split('\n'):
        if l.startswith(attr):
            return l.strip().replace(attr, '')


def fetch_single_label(labels: List[str]) -> Union[str, None]:
    """Fetch a single label for a sentence."""
    for l in labels:
        if l != '*':
            return l


def fetch_pie_idxs(labels: List[str]) -> List[str]:
    """Determine the indices of the PIE components in a sentence."""
    return [str(i) for i, l in enumerate(labels) if l != '*']


def assert_len(
        data: Dict[str, Dict[str, List[str]]],
        split: str,
        *f_names: str):
    """Assert that for every instance the different features have
    the same number of items (e.g. there should be exactly one POS tag
    and label for every token in a sentence).
    """
    lengths = []
    for f_name in f_names:
        lengths.append([len(t) for t in data[split][f_name]])
    for i in range(len(lengths)):
        if i < len(lengths) - 1:
            assert lengths[i] == lengths[i+1]

##################################################
# Build vocab
##################################################


def update_vocab(d: List[List[str]], vocab: Counter) -> None:
    for dp in d:
        vocab.update(dp)


def save_vocab_to_txt(vocab: List[str], out_path: str, file_name: str) -> None:
    with open(join(out_path, file_name), 'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(token + '\n')


##################################################
# Save data
##################################################

def save_data(data: Dict[str, Dict[str, List[str]]], split: str) -> None:
    """Save data."""
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(join('data', split)):
        os.makedirs(join('data', split))

    for k, v in data[split].items():
        with open(join('data', split, k + '.txt'), 'w', encoding='utf-8') as f:
            for i in v:
                if isinstance(i, list):
                    f.write(' '.join(i) + '\n')
                else:
                    f.write(i + '\n')


if __name__ == "__main__":
    args = arg_parser.parse_args()
    corpus_dir = args.corpus_dir

    print("Reading the corpus...")
    sents_per_type, pie_types = read_corpus(corpus_dir)

    print("Splitting the data...")
    split_data, num_instances_per_type = balanced_split(sents_per_type)
    data, label_set = extract_features(split_data)

    print("Saving data...")
    save_data(data, 'train')
    save_data(data, 'dev')
    save_data(data, 'test')

    print("Building vocab and label set...")
    tokens = Counter()
    update_vocab(data['train']['sentences'], tokens)
    update_vocab(data['dev']['sentences'], tokens)
    update_vocab(data['test']['sentences'], tokens)
    tokens = [tok for tok, _ in tokens.items()]
    tokens.insert(0, UNK_WORD)
    tokens.insert(0, PAD_WORD)

    print('Saving vocab and label set...')
    save_vocab_to_txt(tokens, 'data', 'vocab.txt')
    label_set.remove('*')
    label_set = sorted(list(label_set))
    save_vocab_to_txt(label_set, 'data', 'label_set.txt')

    print('Saving PIE type info...')
    write_dict_to_json(
        num_instances_per_type,
        'data',
        'num_instances_per_type.json')

    print("Done.")
