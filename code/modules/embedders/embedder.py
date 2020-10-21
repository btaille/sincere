import torch
from torch import nn

from modules.embedders.word import WordEmbedder
from modules.embedders.char import CharBiLSTMPool
from modules.embedders.bert import BertEmbedder


class Embedder(nn.Module):
    def __init__(self, mode, config, vocab, w_embeddings=None):
        super(Embedder, self).__init__()

        if mode == "word":
            self.embedder = WordEmbedder(vocab.words.val2idx, config.word_embedding_dim, w_embeddings=w_embeddings,
                                         freeze=config.word_freeze, word_dropout=config.word_dropout)

        elif mode == "bert-base":
            self.embedder = BertEmbedder("bert-base-cased", finetune=config.bert_finetune,
                                         word_pooling=config.bert_pool)
        elif mode == "bert-large":
            self.embedder = BertEmbedder("bert-large-cased", finetune=config.bert_finetune,
                                         word_pooling=config.bert_pool)
        elif mode == "char":
            self.embedder = CharBiLSTMPool(vocab.chars.val2idx,
                                           char_embed_dim=config.char_embedding_dim,
                                           char_hidden_dim=config.char_hidden_dim,
                                           pool=config.char_pool,
                                           dropout=config.dropout)
        else:
            print(mode)
            assert False

        self.name = self.embedder.name
        self.w_embed_dim = self.embedder.w_embed_dim

    def forward(self, data):
        output = self.embedder(data)
        return output


class StackedEmbedder(nn.Module):
    def __init__(self, embedders):
        super(StackedEmbedder, self).__init__()
        self.embedders = {embedder.name: embedder for embedder in embedders}
        self.w_embed_dim = 0

        for name, embedder in self.embedders.items():
            self.add_module(name, embedder)
            self.w_embed_dim += embedder.w_embed_dim

    def forward(self, data):
        if len(self.embedders) == 1:
            for name, embedder in self.embedders.items():
                return embedder(data)

        else:
            embeddings = []
            for name, embedder in self.embedders.items():
                embeddings.append(embedder(data)["embeddings"])

            embeddings = torch.cat(embeddings, -1)

            return {"embeddings": embeddings}
