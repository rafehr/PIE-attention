import argparse
import os
from xml.sax.saxutils import prepare_input_source
from collections import Counter
import numpy as np

from data import *
from utils import create_encoders, create_decoders, load_embeddings
from stats import *
from model import *

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--split', default='test',
                    help="Directory containing the data set split")
parser.add_argument('--model_path', help="Path to the model to be evaluated")

if __name__ == '__main__':
    args = parser.parse_args()
    data_path = args.data_path
    split = args.split
    model_path = args.model_path
    vocab_path = join(data_path, 'vocab.txt')
    label_path = join(data_path, 'label_set.txt')

    data = read_data(data_path, split)

    word_to_idx, label_to_idx = create_encoders(vocab_path, label_path)
    idx_to_word, idx_to_label = create_decoders(word_to_idx, label_to_idx)

    enc_data = encode_with(data, word_to_idx, label_to_idx)

    pretrained_weights = load_embeddings(
        len(word_to_idx),
        idx_to_word,
        'embeddings/cc_de_embeddings_300.txt',
        300)

    emb_size = pretrained_weights.shape[1]

    model = AttnModel(
        emb_size=emb_size,
        hidden_size=100,
        pretrained_weights=pretrained_weights,
        out_size=len(label_to_idx),
        p=0.0,
        lstm=True)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    total_preds = torch.tensor([], dtype=torch.long)
    total_true_labels = torch.tensor([], dtype=torch.long)

    total_attn_stats = ([], [], [], [], [], [], [], [], [], [], [], [], [])

    data_iter = data_iterator(enc_data, 32, shuffle=False)
    for batch in data_iter:
        sents, pie_idxs, labels, info = batch
        pos, rels, heads, types, sent_idxs = info
        scores, attn_weights = model(sents, pie_idxs)
        preds = torch.argmax(scores, dim=1)

        dec_sents = decode_sents(sents, idx_to_word)

        pie_noun_idxs, pie_verb_idxs = fetch_pie_pos(
            pos, pie_idxs, dec_sents)

        di_graphs = build_graphs(heads, rels)

        batch_attn_stats = attention_stats(
            info,
            sents,
            idx_to_word,
            pie_idxs,
            attn_weights,
            labels,
            preds,
            pie_noun_idxs,
            pie_verb_idxs,
            di_graphs,
            idx_to_label
        )

        for ts, bs in zip(total_attn_stats, batch_attn_stats):
            ts.extend(bs)

        total_preds = torch.cat((total_preds, preds), dim=0)
        total_true_labels = torch.cat((total_true_labels, labels), dim=0)

    num_instances = len(total_attn_stats[0])

    if not os.path.exists('stats'):
        os.makedirs('stats')

    write_attn_stats_json(total_attn_stats, 'attn_stats.json')

    prec_rec_f1 = precision_recall_fscore_support(
        total_true_labels,
        total_preds,
        average='weighted')

    prec_rec_f1 = [np.round(i * 100, 2) for i in prec_rec_f1[:3]]
    print(prec_rec_f1)
