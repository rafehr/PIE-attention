import json
import os
import torch
from torch import Tensor
import numpy as np
from typing import Tuple, List, Dict, Any


def load_embeddings(
    vocab_size: int,
    idx_to_word: Dict[int, str],
    emb_path: str,
    emb_dim: int
) -> Tensor:
    """Loads pretrained fasttext embeddings from disk."""
    emb_dict = {}
    with open(emb_path, 'r', encoding='utf-8') as f:
        embs = [e.split() for e in f.read().strip().split('\n')]
        for e in embs:
            emb_dict[e[0]] = np.asarray(e[1:], dtype='float32')

    pretrained_weights = np.zeros((vocab_size, emb_dim), dtype='float32')

    count = 0
    for i, tok in idx_to_word.items():
        if i != 0:
            try:
                pretrained_weights[i] = emb_dict[tok]
            except KeyError as e:
                print(tok)
                count += 1
                print(count)

    return torch.from_numpy(pretrained_weights)


def write_dict_to_json(dict: Dict, out_path: str, file_name: str) -> None:
    """Saves a dictionary to a JSON-file."""
    file_path = os.path.join(out_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict, f, indent=4, ensure_ascii=False)


def assert_equal_len(*lists: List[Any]) -> None:
    assert len(set([len(l) for l in lists])) == 1


def create_encoder(data_path: str) -> Dict[str, int]:
    with open(data_path, 'r', encoding='utf-8') as f:
        cls = f.read().strip().split('\n')
    cls_to_idx = {c: i for i, c in enumerate(cls)}
    return cls_to_idx


def create_decoder(cls_to_idx: Dict[str, int]) -> Dict[int, str]:
    idx_to_cls = {i: t for t, i in cls_to_idx.items()}
    return idx_to_cls


def create_encoders(
    vocab_path: str,
    label_path: str
) -> Tuple[Dict[str, int], Dict[str, int]]:
    word_to_idx = create_encoder(vocab_path)
    label_to_idx = create_encoder(label_path)
    return word_to_idx, label_to_idx


def create_decoders(
    word_to_idx: Dict[str, int],
    label_to_idx: Dict[str, int]
) -> Tuple[Dict[int, str], Dict[int, str]]:
    idx_to_word = create_decoder(word_to_idx)
    idx_to_label = create_decoder(label_to_idx)
    return idx_to_word, idx_to_label


def decode_sents(
    sents: Tensor,
    idx_to_word: Dict[int, str]
) -> List[List[str]]:
    dec_sents = [
        [idx_to_word[i.item()] for i in sent]
        for sent in sents
    ]
    return dec_sents


def write_json(dict, dir_path, file_name):
    file_path = os.path.join(dir_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict, f, indent=4, ensure_ascii=False)
