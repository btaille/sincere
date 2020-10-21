import numpy as np
import torch
from torch import nn
from utils.rnn_utils import PackedRNN

from torch.nn.modules.pooling import MaxPool2d, AvgPool2d


class CharBiLSTMPool(nn.Module):
    def __init__(self, char2idx, char_embed_dim=100, char_hidden_dim=25, pool="last", dropout=0., init_std=0.001):
        super().__init__()
        assert pool in ["max", "avg", "last"], "'{}' should be max, avg or last.".format(pool)

        self.char2idx = char2idx
        self.char_hidden_dim = char_hidden_dim
        self.char_embed_dim = char_embed_dim
        self.w_embed_dim = 2 * char_hidden_dim

        self.pool = pool

        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)

        if self.pool == "max":
            self.pool_fn = MaxPool2d
        elif self.pool == "avg":
            self.pool_fn = AvgPool2d

        self.char_embeddings = nn.Embedding(len(self.char2idx), self.char_embed_dim)
        self.char_embeddings.weight.data.normal_(mean=0.0, std=init_std)

        char_lstm = nn.LSTM(self.char_embed_dim, self.char_hidden_dim, bidirectional=True, batch_first=True)
        self.packed_char_lstm = PackedRNN(char_lstm)
        self.name = "char"

    @property
    def device(self):
        return self.char_embeddings.weight.device

    def forward(self, data, char_ids_k="char_ids", n_chars_k="n_chars"):
        char_embeddings = self.drop(self.char_embeddings(data[char_ids_k].to(self.device, non_blocking=True)))
        batch_size, max_nwords, max_nchars, char_embedding_dim = char_embeddings.size()

        # Flatten for LSTM
        # [batch, max words, max chars, char embedding] -> [batch * max words, max chars, char embedding]
        flat_embeddings = char_embeddings.view(-1, max_nchars, char_embedding_dim)
        flat_lens = np.concatenate(data[n_chars_k])

        # LSTM
        flat_out, char_hidden = self.packed_char_lstm(flat_embeddings, flat_lens)

        if self.pool in ["max", "avg"]:
            # Pooling
            pooled_flat_out = self.pool_fn([flat_out.size(1), 1])(flat_out).view(-1, 2 * self.char_hidden_dim)
        elif self.pool == "last":
            # Unpack and pad last hidden states
            pooled_flat_out = torch.cat((char_hidden[0][0], char_hidden[0][1]), -1)

        # Unflatten
        return {"embeddings": pooled_flat_out.view(batch_size, -1, 2 * self.char_hidden_dim)}
