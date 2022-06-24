import argparse
import spacy
from typing import List
from spacy.tokens import Doc
from spacy.lang.de import German
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")


def custom_tokenizer(text: str) -> Doc:
    """A custom tokenizer that just uses Python's split method to
    "tokenize" a sentence, since the sentences were arlready tokenized.
    """
    return Doc(nlp.vocab, text.split())


def load_dataset(split_dir: str) -> List[str]:
    with open(join(split_dir, 'sentences.txt'), 'r', encoding='utf-8') as f:
        sents = [s for s in f.read().strip().split('\n')]
    return sents


def parse_sent(spacy_pipe: German, sent: str) -> Doc:
    """Parses a sentence and returns a Doc object from which the
    features can be extracted.
    """
    parsed_sent = spacy_pipe(sent)
    tokens = [tok.text for tok in parsed_sent]
    assert len(tokens) == len(sent.split())
    try:
        assert len(' '.join(tokens)) == len(sent)
    except AssertionError:
        print(sent)
        print(' '.join(tokens))
    return parsed_sent


def save_feature(feature: List[str], split: str, f_name: str) -> None:
    with open(join(data_dir, split, f_name), 'w', encoding='utf-8') as f:
        for s in feature:
            f.write(' '.join(s) + '\n')


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir

    nlp = spacy.load('de_dep_news_trf')
    nlp.tokenizer = custom_tokenizer

    for split in ['train', 'dev', 'test']:
        sents = load_dataset(join(data_dir, split))
        pos_data, deprel_data, head_data = [], [], []
        for s in sents:
            pos, deprels, heads = [], [], []
            for tok in parse_sent(nlp, s):
                pos.append(tok.pos_)
                deprels.append(tok.dep_)
                heads.append(str(tok.head.i))
            assert len(pos) == len(deprels) == len(heads)
            pos_data.append(pos)
            deprel_data.append(deprels)
            head_data.append(heads)
        assert len(pos_data) == len(deprel_data) == len(head_data)

        # save_feature(pos_data, split, 'tiger_pos.txt')
        save_feature(deprel_data, split, 'deprels.txt')
        save_feature(head_data, split, 'heads.txt')
