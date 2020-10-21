import torch
import torch.nn as nn
from data.utils import mask


class WordDropout(nn.Module):
    def __init__(self, p, unknown=1, pad=0):
        super(WordDropout, self).__init__()
        self.p = p
        self.unknown = unknown
        self.pad = pad

    def forward(self, word_ids, nwords):
        if self.training:
            pad_mask = mask(nwords, device=word_ids.device)
            drop_mask = word_ids.bernoulli(self.p).to(torch.bool)

            word_ids = word_ids.masked_fill(drop_mask, self.unknown)
            word_ids = word_ids.masked_fill(pad_mask == 0, self.pad)
        return word_ids


class WordEmbedder(nn.Module):
    def __init__(self, word2idx, embed_dim, w_embeddings=None, freeze=False, word_dropout=0., init_std=0.001):
        super(WordEmbedder, self).__init__()

        self.word2idx = word2idx
        self.w_embed_dim = embed_dim
        self.freeze_embeddings = freeze
        self.name = "word"

        self.word_dropout = word_dropout
        self.word_drop = WordDropout(self.word_dropout)

        # layers
        if w_embeddings is None:
            w_embeddings = torch.zeros(len(self.word2idx), self.w_embed_dim)
            w_embeddings.normal_(mean=0.0, std=init_std)
            assert not self.freeze_embeddings
        else:
            assert w_embeddings.size(1) == self.w_embed_dim

        self.word_embeddings = nn.Embedding(w_embeddings.size(0), w_embeddings.size(1))
        self.word_embeddings.weight = nn.Parameter(w_embeddings)

        if self.freeze_embeddings:
            self.word_embeddings.weight.requires_grad = False

    @property
    def device(self):
        return self.word_embeddings.weight.device

    def forward(self, data, word_ids_k="word_ids", n_words_k="n_words"):
        # Word embeddings
        embeds = self.word_embeddings(
            self.word_drop(data[word_ids_k].to(self.device, non_blocking=True), data[n_words_k]))

        return {"embeddings": embeds}
