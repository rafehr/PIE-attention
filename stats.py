import argparse
import os
from utils import *
from collections import Counter, defaultdict
import networkx as nx
import json
import numpy as np
from data import *
from scipy.stats import pearsonr

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--stats_path',
    default='stats',
    help="Path to the directory containing the attention statistics files."
)
arg_parser.add_argument(
    '--split_dir',
    default='test',
    help="The directory containing a specific split of the data set."
)
arg_parser.add_argument(
    '--stats_filename',
    default='attn_stats.json',
    help="Path to the file containing the attention statistics."
)


def fetch_max_attn_idxs(attn_weights):
    max_attn_idxs = [torch.argmax(w, dim=0).item() for w in attn_weights]
    return max_attn_idxs


def pie_pos(pos, pie_idxs):
    return [[(int(i), p[i]) for i in idxs] for p, idxs in zip(pos, pie_idxs)]


def fetch_max_attn_head_idxs(heads, max_attn_idxs):
    return [int(h[i]) for h, i in zip(heads, max_attn_idxs)]


def fetch_max_attn_heads(dec_sents, max_attn_head_idxs):
    return [s[i] for s, i in zip(dec_sents, max_attn_head_idxs)]


def fetch_max_attn_words(dec_sents, max_attn_idxs):
    return [s[i] for s, i in zip(dec_sents, max_attn_idxs)]


def check_for_pie_heads(max_attn_heads, pie_idxs):
    return [h in i for h, i in zip(max_attn_heads, pie_idxs)]


def fetch_head_pos(max_attn_heads, pos):
    return [p[h] for h, p in zip(max_attn_heads, pos)]


def fetch_attn_word_pos(pos, max_attn_idxs):
    return [p[i] for p, i in zip(pos, max_attn_idxs)]


def fetch_max_attn_rels(rels, max_attn_idxs):
    return [r[i] for r, i in zip(rels, max_attn_idxs)]


def fetch_max_attn_scores(attn_weights):
    return [np.round(s.item(), 2) for s in torch.max(attn_weights, dim=1)[0]]


def distance_to_pie(max_attn_idxs, idxs):
    return [abs(ai - i) for ai, i in zip(max_attn_idxs, idxs)]


def fetch_pie_pos(pos, pie_idxs, dec_sents):
    pie_pos = [
        [(ps[i.item()], s[i.item()], i.item()) for i in pi]
        for ps, pi, s in zip(pos, pie_idxs, dec_sents)
    ]
    pie_noun_idxs, pie_verb_idxs = [], []
    pie_noun_toks, pie_verb_toks = [], []

    for ps, ds in zip(pie_pos, dec_sents):
        noun = False
        verb = False
        for p, t, i in ps:
            if p in ['NN']:
                pie_noun_idxs.append(i)
                pie_noun_toks.append((i, p, t))
                noun = True
            if p not in ['NN', 'APPR', 'APPRART']:
                pie_verb_idxs.append(i)
                pie_verb_toks.append((i, p, t))
        if noun == False:
            print(ds)
    try:
        assert len(pie_noun_idxs) == len(pie_verb_idxs)
    except AssertionError:
        # print(len(pie_noun_idxs))
        # print(len(pie_verb_idxs))
        # print(dec_sents)
        # print(pie_noun_toks)
        # print(pie_verb_toks)
        # # print(pie_noun_idxs)
        # # print(pie_verb_idxs)
        # for d in dec_sents:
        #     print(' '.join(d))
        pie_verb_idxs.pop()
    return pie_noun_idxs, pie_verb_idxs


def build_graphs(heads, rels):
    di_graphs = []
    for h, r in zip(heads, rels):
        assert len(h) == len(r)
        dg = nx.DiGraph()
        dg.add_nodes_from(range(len(h)))
        edges = [(int(head), i) for i, head in enumerate(h)]
        relations = [{'Rel': relation} for relation in r]
        named_edges = [(e[0], e[1], rel) for e, rel in zip(edges, relations)]
        dg.add_edges_from(named_edges)
        di_graphs.append(dg)
    return di_graphs


def fetch_path(max_attn_idxs, idxs, di_graphs, dec_sents):
    paths = []
    for mai, i, dg, sent in zip(max_attn_idxs, idxs, di_graphs, dec_sents):
        try:
            path = nx.shortest_path(dg, i, mai)
            all_rels = []
            if len(path) > 1:
                rel = dg[path[0]][path[1]]['Rel']
                pie_head = True if len(path) == 2 else False
                for v_i, _ in enumerate(path):
                    all_rels.append(dg[path[v_i]][path[v_i+1]]['Rel'])
                    if v_i == (len(path) - 2):
                        break
            else:
                rel = 'identity'
                pie_head = 'identical'
                all_rels = ['identity']
            path_tokens = [sent[i] for i in path]
            paths.append(
                ('directed', path, len(path) - 1, path_tokens,
                 rel, all_rels, pie_head)
            )
        except nx.exception.NetworkXNoPath:
            dg = dg.to_undirected()
            try:
                path = nx.shortest_path(dg, i, mai)
                all_rels = []
                if len(path) > 1:
                    rel = dg[path[0]][path[1]]['Rel']
                    pie_head = True if len(path) == 2 else False
                    all_rels = []
                    for v_i, _ in enumerate(path):
                        all_rels.append(dg[path[v_i]][path[v_i+1]]['Rel'])
                        if v_i == (len(path) - 2):
                            break
                else:
                    rel = 'identity'
                    pie_head = 'identical'
                    all_rels = ['identity']
            except nx.exception.NetworkXNoPath:
                path = []
                rel = 'no path'
                pie_head = False
                all_rels = []
            path_tokens = [sent[i] for i in path]
            paths.append(
                ('undirected', path, len(path) - 1, path_tokens,
                 rel, all_rels, pie_head)
            )
    try:
        assert len(paths) == len(max_attn_idxs)
    except AssertionError:
        pass
        # print(paths)
        # print(max_attn_idxs)
    try:
        assert len({len(p) for p in paths}) == 1
    except AssertionError:
        pass
        # print(paths)
        # print(max_attn_idxs)
    return paths


def write_attn_stats_json(attn_stats, file_name):
    f = open(file_name, 'w', encoding='utf-8')
    sent_ids, sents, attn_words, attn_scores, attn_words_pos, attn_weights,\
        preds, labels, path_to_verb, path_to_noun, verb_dist, noun_dist, types = attn_stats

    i = 0
    attn_stats_dict = {}
    for sent_id, sent, attn_word, attn_score, attn_word_pos, attn_weight,\
        pred, label, ptv, ptn, vd, nd, t in zip(
            sent_ids, sents, attn_words, attn_scores, attn_words_pos, attn_weights,
            preds, labels, path_to_verb, path_to_noun, verb_dist, noun_dist,
            types):
        verb_direction, verb_path, verb_path_dist, verb_path_tokens,\
            verb_rel, verb_path_rels, verb_pie = ptv
        noun_direction, noun_path, noun_path_dist, noun_path_tokens,\
            noun_rel, noun_path_rels, noun_pie = ptn
        attn_stats_dict[i] = {
            'sentence_id': sent_id,
            'sentence': ' '.join(sent),
            'attention_word': attn_word,
            'attention_score': attn_score,
            'attention_POS': attn_word_pos,
            'PIE_verb_direction': verb_direction,
            'PIE_verb_path': ' -> '.join([str(a) for a in verb_path]),
            'PIE_verb_path_tokens': ' -> '.join(verb_path_tokens),
            'PIE_verb_path_rels': ' -> '.join(verb_path_rels),
            'PIE_verb_path_dist': verb_path_dist,
            'PIE_verb_relation': verb_rel,
            'PIE_verb_head': verb_pie,
            'PIE_noun_direction': noun_direction,
            'PIE_noun_path': ' -> '.join([str(a) for a in noun_path]),
            'PIE_noun_path_tokens': ' -> '.join([str(a) for a in noun_path_tokens]),
            'PIE_noun_path_rels': ' -> '.join([str(a) for a in noun_path_rels]),
            'PIE_noun_path_dist': noun_path_dist,
            'PIE_noun_relation': noun_rel,
            'PIE_noun_head': noun_pie,
            'PIE_verb_word_distance': vd,
            'PIE_noun_word_distance': nd,
            'attention_scores': attn_weight,
            'predicted_label': pred,
            'true_label': label,
            'PIE type': t
        }
        i += 1

    f.close()

    print()

    write_json(attn_stats_dict, 'stats', file_name)


def write_attn_stats(attn_stats):
    f = open('attn_stats.tsv', 'w', encoding='utf-8')
    sents, words, scores, pos_tags, heads, pie_heads,\
        head_pos_tags, distances, rels, weights, preds, labels = attn_stats

    for sent, word, score, pos, head, pie_head, head_pos, dis, rel, weight, pred, lab in zip(
        sents, words, scores, pos_tags, heads, pie_heads, head_pos_tags, distances, rels, weights, preds, labels
    ):
        f.write(
            '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                ' '.join(sent),
                word,
                score,
                pos,
                head,
                pie_head,
                head_pos,
                str(dis),
                rel,
                ' '.join(weight),
                pred,
                lab
            )
        )
    f.close()


def filter_sents(dec_sents, attn_weights, pos, target_pos, sent_idxs):
    filtered_ids = []
    filtered_sents = []
    max_attn_idxs = fetch_max_attn_idxs(attn_weights)
    assert len(max_attn_idxs) == len(pos)
    for i, p, sent, sent_i in zip(max_attn_idxs, pos, dec_sents, sent_idxs):
        sent = [t for t in sent if t != 'UNK']
        if p[i] in target_pos:
            tagged_sent = [
                f'<attn>{t}</attn>' if idx == i else t
                for idx, t in enumerate(sent)
            ]
            filtered_sents.append((sent_i, tagged_sent))
            filtered_ids.append(sent_i)
    return filtered_ids, filtered_sents


def write_features(feature, out_dir, feature_name):
    out_path = os.path.join(out_dir, feature_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        for i in feature:
            if isinstance(i, list):
                f.write(' '.join(i) + '\n')
            else:
                f.write(i + '\n')


def filter_data(eval_data, filtered_ids, out_dir):
    tokens, labels, pos, deprels, heads,\
        single_labels, types, sent_idxs = [], [], [], [], [], [], [], []
    for t, l, p, d, h, sl, ty, i in eval_data:
        if i in filtered_ids:
            tokens.append(t)
            labels.append(l)
            pos.append(p)
            deprels.append(d)
            heads.append(h)
            single_labels.append(sl)
            types.append(ty)
            sent_idxs.append(i)
    write_features(tokens, out_dir, 'tokens.txt')
    write_features(labels, out_dir, 'labels.txt')
    write_features(pos, out_dir, 'pos.txt')
    write_features(deprels, out_dir, 'deprels.txt')
    write_features(heads, out_dir, 'heads.txt')
    write_features(single_labels, out_dir, 'single_labels.txt')
    write_features(types, out_dir, 'types.txt')
    write_features(sent_idxs, out_dir, 'idxs.txt')


def attention_stats(
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
        idx_to_label):

    pos, rels, heads, types, sent_ids = info

    dec_sents = decode_sents(sents, idx_to_word)
    max_attn_idxs = fetch_max_attn_idxs(attn_weights)
    max_attn_words = fetch_max_attn_words(dec_sents, max_attn_idxs)
    max_attn_pos = fetch_attn_word_pos(pos, max_attn_idxs)
    max_attn_scores = fetch_max_attn_scores(attn_weights)

    path_to_verb = fetch_path(
        max_attn_idxs, pie_verb_idxs, di_graphs, dec_sents)
    path_to_noun = fetch_path(
        max_attn_idxs, pie_noun_idxs, di_graphs, dec_sents)

    max_attn_head_idxs = fetch_max_attn_head_idxs(heads, max_attn_idxs)
    max_attn_heads = fetch_max_attn_heads(dec_sents, max_attn_head_idxs)
    max_attn_rels = fetch_max_attn_rels(rels, max_attn_idxs)
    head_pos = fetch_head_pos(max_attn_head_idxs, pos)
    pie_heads = check_for_pie_heads(max_attn_head_idxs, pie_idxs)

    verb_distance = distance_to_pie(max_attn_idxs, pie_verb_idxs)
    noun_distance = distance_to_pie(max_attn_idxs, pie_noun_idxs)

    # with open('data/single_labels_vocab.txt', 'r') as f:
    #     single_labels = f.read().strip().split('\n')
    #     idx_to_label = {i: l for i, l in enumerate(single_labels)}

    labels = [idx_to_label[i] for i in labels.tolist()]
    preds = [idx_to_label[i] for i in preds.tolist()]

    attn_weights = [
        [str(w[0]) for w in aw.detach().numpy()]
        for aw in attn_weights
    ]

    stats = (
        sent_ids,
        dec_sents,
        max_attn_words,
        max_attn_scores,
        max_attn_pos,
        attn_weights,
        preds,
        labels,
        path_to_verb,
        path_to_noun,
        verb_distance,
        noun_distance,
        types
    )
    return stats


def read_attn_file(stats_path):
    with open(stats_path, 'r', encoding='utf-8') as s:
        attn_stats = [l.split('\t') for l in s.read().strip().split('\n')]
    return attn_stats


def stats_to_dict(attn_stats):
    i = 0
    attn_dict = {}
    for a in attn_stats:
        sent, attn_word, attn_score, attn_POS, attn_head,\
            PIE_head, head_POS, distance, rel, attn_scores, pred, label = a
        attn_dict[i] = {
            'sentence': sent,
            'attention_word': attn_word,
            'attention_score': attn_score,
            'attention_POS': attn_POS,
            'attention_head': attn_head,
            'PIE_head': PIE_head,
            'head_POS': head_POS,
            'distance_attn_word_head': distance,
            'relation_attn_word_head': rel,
            'attention_scores': attn_scores,
            'predicted_label': pred,
            'true_label': label
        }
        i += 1
    return attn_dict


def compute_ratio(dict, feature, m, top):
    count = Counter([s[feature] for s in dict.values()])
    ratios = {i: round((v/m) * 100, 2) for i, v in count.most_common(top)}
    return ratios


def compute_feat_ratio(feats, value):
    rel_counts = Counter(feats)
    return round((rel_counts[value]/len(feats)) * 100, 2)


def len_vs_feature(stats, feature, value):
    d = defaultdict(list)
    for k, v in stats.items():
        sent = [t for t in stats[k]['sentence'].split() if t != 'UNK']
        len_sent = len(sent)
        feat = v[feature]
        d[len_sent].append(feat)
    return compute_over_intervall(d, value)


def compute_over_intervall(len_vs_feat, value):
    feats, d = [], {}
    t = 10
    for k, v in len_vs_feat.items():
        if k <= t:
            feats.extend(v)
        else:
            r = compute_feat_ratio(feats, value)
            d[f'<={t}'] = (r, f'#sents: {len(feats)}')
            t += 10
            feats = []
            feats.extend(v)
    return d


def filter_ocs(stats):
    new_rels = []
    for k, v in stats.items():
        if v['PIE_verb_relation'] == 'oc':
            rels = v['PIE_verb_path_rels'].split(' -> ')
            for r in rels:
                if r != 'oc':
                    new_rels.append(r)
                    v['PIE_verb_relation'] = r
                    break
    return new_rels


def compute_overall_stats(stats, top=5):
    m = len(stats)
    distr_max_attn_word_pos = compute_ratio(
        stats, 'attention_POS', m, top
    )
    pie_verb_path_distance = compute_ratio(
        stats, 'PIE_verb_path_dist', m, top
    )
    pie_verb_relation = compute_ratio(
        stats, 'PIE_verb_relation', m, top
    )
    pie_noun_path_distance = compute_ratio(
        stats, 'PIE_noun_path_dist', m, top
    )
    pie_noun_relation = compute_ratio(
        stats, 'PIE_noun_relation', m, top
    )
    pie_verb_word_distance = compute_ratio(
        stats, 'PIE_verb_word_distance', m, top
    )
    pie_noun_word_distance = compute_ratio(
        stats, 'PIE_noun_word_distance', m, top
    )
    pie_verb_direction = compute_ratio(
        stats, 'PIE_verb_direction', m, top
    )
    pie_noun_direction = compute_ratio(
        stats, 'PIE_noun_direction', m, top
    )
    max_attn_scores = []

    pie_verb_attn_word = []
    pie_noun_attn_word = []

    for i in range(m):
        max_attn_scores.append(stats[str(i)]['attention_score'])
        if stats[str(i)]['PIE_verb_relation'] == 'identity':
            pie_verb_attn_word.append(stats[str(i)]['PIE_verb_relation'])
        if stats[str(i)]['PIE_noun_relation'] == 'identity':
            pie_noun_attn_word.append(stats[str(i)]['PIE_noun_relation'])

    mean_max_attn_score = np.mean(np.array(max_attn_scores))

    # Compute the ratio of the PIE verb being the attention word
    pie_verb_attn_word_ratio = (len(pie_verb_attn_word)/m) * 100
    # Compute the ratio of the PIE noun being the attention word
    pie_noun_attn_word_ratio = (len(pie_noun_attn_word)/m) * 100
    # Compute ratio of a certain feature with regard to sentence length
    len_vs_sb = len_vs_feature(stats, 'PIE_verb_relation', 'sb')
    len_vs_oa = len_vs_feature(stats, 'PIE_verb_relation', 'oa')
    len_vs_mo = len_vs_feature(stats, 'PIE_verb_relation', 'mo')
    len_vs_nn = len_vs_feature(stats, 'attention_POS', 'NN')
    len_vs_adja = len_vs_feature(stats, 'attention_POS', 'ADJA')
    len_vs_ne = len_vs_feature(stats, 'attention_POS', 'NE')
    attn_stats = {
        "POS distribution": distr_max_attn_word_pos,
        "Path distance to PIE verb": pie_verb_path_distance,
        "Word distance to PIE verb": pie_verb_word_distance,
        "Relation to PIE verb (first arc)": pie_verb_relation,
        "Direction of path between PIE verb and attention word": pie_verb_direction,
        "PIE verb equal to attention word": np.round(pie_verb_attn_word_ratio, 2),
        "Path distance to PIE noun": pie_noun_path_distance,
        "Word distance to PIE noun": pie_noun_word_distance,
        "Relation ot PIE noun (first arc)": pie_noun_relation,
        "Direction of path between PIE noun and attention word": pie_noun_direction,
        "PIE noun equal to attention word": np.round(pie_noun_attn_word_ratio, 2),
        "Ratio of subject relation wrt. length": len_vs_sb,
        "Ratio of object relation wrt. length": len_vs_oa,
        "Ratio of modifier relation wrt. length": len_vs_mo,
        "Ratio of NN POS wrt. length": len_vs_nn,
        "Ratio of ADJA POS wrt. length": len_vs_adja,
        "Ratio of NE POS relation wrt. length":  len_vs_ne,
        "Max attn score mean and std deviation": (np.mean(max_attn_scores), np.std(max_attn_scores))
    }
    return attn_stats


if __name__ == '__main__':
    args = arg_parser.parse_args()
    stats_path = args.stats_path
    split = args.split_dir
    stats_filename = args.stats_filename
    stats_file_path = os.path.join(stats_path, split, stats_filename)

    # Open file with individual properties of sentences
    with open(stats_file_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    # # Filter out object clause relations
    no_ocs_file = 'attn_stats_no_ocs_II.json'
    new_rels = filter_ocs(stats)
    print(Counter(new_rels))

    write_json(
        stats,
        os.path.join(stats_path, split),
        no_ocs_file
    )

    def get_str_len(my_string: str) -> int:
        return len(my_string)

    get_str_len(5)

    exit()

    distances = []
    attn_scores = []
    for k, v in stats.items():
        distances.append(v['PIE_verb_word_distance'])
        attn_scores.append(v['attention_score'])
    print(pearsonr(distances, attn_scores))
    print(np.histogram(attn_scores, bins=10))

    m = len(stats)

    # Creating a dictionary per PIE type
    stats_per_type = {}
    for i in range(m):
        if stats[str(i)]["PIE type"] not in stats_per_type:
            stats_per_type[stats[str(i)]["PIE type"]] = [stats[str(i)]]
        else:
            stats_per_type[stats[str(i)]["PIE type"]].append(stats[str(i)])

    for key, value in stats_per_type.items():
        stats_per_type[key] = {str(i): v for i, v in enumerate(value)}

    # Creating a dictionary per predicted label
    stats_per_label = {}
    for i in range(m):
        if stats[str(i)]["predicted_label"] not in stats_per_label:
            stats_per_label[stats[str(i)]["predicted_label"]] = [stats[str(i)]]
        else:
            stats_per_label[stats[str(i)]["predicted_label"]].append(
                stats[str(i)])

    for key, value in stats_per_label.items():
        stats_per_label[key] = {str(i): v for i, v in enumerate(value)}

    attn_stats = compute_overall_stats(stats, top=5)
    print(attn_stats)

    write_json(attn_stats, os.path.join(stats_path, split), 'dist.json')
    # exit()

    for key, value in attn_stats.items():
        try:
            r = ['{}: {}\%'.format(k, str(v)) for k, v in value.items()]
            tr = key + ' & ' + ' & '.join(r) + ' \\\\'
            print('\\hline')
            print(tr)
        except AttributeError:
            continue
