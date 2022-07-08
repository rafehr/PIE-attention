import fasttext
import fasttext.util
import numpy as np
import os

import ntpath
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--model_path', help='Path to fasttext model.')
arg_parser.add_argument('--vocab_path', help='Path to the vocabulary file.')
arg_parser.add_argument('--num_dims', default=300, type=int,
                        help='Number of dimensions the embeddings should be reduced to.')

if __name__ == '__main__':
    args = arg_parser.parse_args()
    vocab_path = args.vocab_path
    num_dims = args.num_dims
    model_path = args.model_path

    ft = fasttext.load_model(model_path)
    emb_dict = {}

    if num_dims != 300:
        fasttext.util.reduce_model(ft, num_dims)

    with open(vocab_path, 'r', encoding='utf-8') as vocab:
        for word in vocab:
            word = word.strip()
            emb = ' '.join([str(i) for i in ft.get_word_vector(word)])
            emb_dict[word] = emb

    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')

    model_name = ntpath.basename(model_path).split('.')
    out_file_name = '{}_{}_{}_{}.txt'.format(
        model_name[0], model_name[1], 'embeddings', num_dims)
    out_file_path = os.path.join('embeddings', out_file_name)

    with open(out_file_path, 'w', encoding='utf-8') as emb_file:
        for word in emb_dict:
            emb_file.write(word + ' ' + emb_dict[word] + '\n')
