from unicodedata import bidirectional
import torch
import torch.nn as nn
import math


class AttnModel(nn.Module):
    def __init__(
        self,
        emb_size,
        hidden_size,
        pretrained_weights,
        out_size,
        p,
        lstm
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(
            pretrained_weights,
            freeze=True,
            padding_idx=0)
        self.dropout = nn.Dropout(p)
        if lstm == True:
            self.cont = nn.LSTM(
                emb_size,
                hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.2
            )
        else:
            self.cont = nn.GRU(
                emb_size,
                hidden_size,
                batch_first=True,
                bidirectional=True
            )
        self.attn = AdditiveAttention(
            query_size=emb_size,
            key_size=(hidden_size * 2),
            hidden_size=hidden_size)
        self.linear_1 = nn.Linear(
            emb_size + (hidden_size * 2),
            hidden_size
        )
        self.linear_2 = nn.Linear(hidden_size, out_size)

    def forward(self, inp, pie_idxs):
        pad_mask = inp == 0
        embs = self.embedding(inp)

        pie_embs = [embs[b][p] for b, p in enumerate(pie_idxs)]
        queries = torch.stack(
            [torch.mean(embs, dim=0) for embs in pie_embs],
            dim=0
        ).unsqueeze(1)

        embs = self.dropout(embs)

        hiddens, _ = self.cont(embs)

        attn_weights, context = self.attn(queries, hiddens, pad_mask)

        concat = torch.cat(
            (context.transpose(2, 1), queries),
            dim=2
        )
        hidden_layer = torch.tanh(self.linear_1(concat))
        out = self.linear_2(hidden_layer).squeeze(1)

        return out, attn_weights


class AdditiveAttention(nn.Module):
    def __init__(self, query_size, key_size, hidden_size):
        super().__init__()

        self.W_q = nn.Linear(query_size, hidden_size, bias=False)
        self.W_k = nn.Linear(key_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, queries, keys, pad_mask):
        trans_queries = self.W_q(queries)
        trans_keys = self.W_k(keys)

        features = torch.tanh(trans_queries + trans_keys)

        scores = self.w_v(features)
        scores[pad_mask] = -float('inf')
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(keys.transpose(2, 1), self.dropout(attn_weights))
        return attn_weights, context
