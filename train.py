import argparse
import torch
import torch.nn as nn
from model import AttnModel
from data import data_iterator, read_data, encode_with
import os
from os.path import join
from utils import create_encoders, create_decoders, load_embeddings

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data',
                    help='Directory containing the dataset')
parser.add_argument('--model_name', help='Name of the model')

if __name__ == '__main__':
    args = parser.parse_args()
    data_path = args.data_path
    model_name = args.model_name
    vocab_path = join(data_path, 'vocab.txt')
    label_path = join(data_path, 'label_set.txt')

    data = read_data(data_path, 'train')

    word_to_idx, label_to_idx = create_encoders(vocab_path, label_path)
    idx_to_word, idx_to_label = create_decoders(word_to_idx, label_to_idx)

    enc_data = encode_with(data, word_to_idx, label_to_idx)

    pretrained_embs = load_embeddings(
        len(word_to_idx),
        idx_to_word,
        'embeddings/cc_de_embeddings_300.txt',
        300
    )

    emb_size = pretrained_embs.shape[1]

    model = AttnModel(
        emb_size=emb_size,
        hidden_size=100,
        pretrained_weights=pretrained_embs,
        out_size=len(label_to_idx),
        p=0.0,
        lstm=True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for e in range(2):
        total_loss = 0
        data_iter = data_iterator(enc_data, 32)
        for batch in data_iter:
            sents, pie_idxs, labels, info = batch
            scores, attn_weights = model(sents, pie_idxs)
            batch_loss = loss(scores, labels)
            total_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("Epoch: {}, total loss: {}".format(e+1, total_loss))

    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')

    torch.save(
        model.state_dict(),
        os.path.join('trained_models', model_name)
    )
