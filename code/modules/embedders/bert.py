import logging

import torch
from torch import nn
from torch.nn import LayerNorm
from transformers import BertModel, BertTokenizer

from data.utils import mask


def compute_bert_alignment(original_tokens, bert_tokens):
    """
    Compute alignments between original word tokenization and BERT tokenization
    """

    indices = []

    i, j = 0, 0
    current_token = ""

    while i < len(original_tokens):
        if bert_tokens[j] == "[UNK]":
            logging.info("[UNK] in BERT tokenize")
            return [1] * len(original_tokens), [i for i in range(len(original_tokens))]

        if current_token == "":
            current_m = j

        if bert_tokens[j][:2] == "##":
            current_token += bert_tokens[j][2:]
        else:
            current_token += bert_tokens[j]

        if original_tokens[i] == current_token:
            indices.append(current_m)
            i += 1
            current_token = ""

        j += 1

    # Compute a mask tensor of size (n_words, n_tokens) which maps original with corresponding tokens
    alignment_mask = torch.zeros((len(original_tokens), len(bert_tokens)), dtype=torch.bool)
    for word_id, start in enumerate(indices):
        if word_id < len(indices) - 1:
            end = indices[word_id + 1]
            alignment_mask[word_id, start:end] = 1
        else:
            alignment_mask[word_id, start:] = 1

    return indices, alignment_mask


def pad(sequences, pad=0):
    maxlen = max([len(seq) for seq in sequences])
    return [seq + [pad] * (maxlen - len(seq)) for seq in sequences]


def bert_tokenize(sample, tokenizer, device="cpu"):
    """
    sample with minimal keys "words" and "n_words"
        sample["words"] : list of words
        sample["n_words"] : int number of words
    tokenizer = BertTokenizer
    """

    tokenized_text = tokenizer.tokenize(" ".join(sample["words"]))
    indices, alignment_mask = compute_bert_alignment(sample["words"][:sample["n_words"]], tokenized_text)

    tokenized_text = ["[CLS]"] + tokenized_text + ["[SEP]"]
    indices = [i + 1 for i in indices]

    n_words, n_tokens = alignment_mask.size()
    zeros = torch.zeros((n_words, 1), dtype=torch.bool, device=device)
    alignment_mask = torch.cat([zeros, alignment_mask.to(device), zeros], dim=1)

    output = {"bert_tokens": tokenized_text,
              "bert_token_ids": torch.tensor(tokenizer.convert_tokens_to_ids(tokenized_text)).to(device),
              "n_bert_tokens": len(tokenized_text),
              "bert_indices": torch.tensor(indices).to(device), "bert_alignment": alignment_mask}

    return output


def token2word_embeddings(data, pooling="max"):
    """Pool subword bert embeddings into word embeddings"""
    assert pooling in ["first", "max", "sum", "avg"]

    if pooling == "first":
        # embeddings (bs, max_n_tokens, h_dim)
        embeddings = data["bert_embeddings"]
        indices = data["bert_indices"].long().to(embeddings.device)
        indices = indices.unsqueeze(-1).repeat(1, 1, embeddings.size(-1))
        return embeddings.gather(1, indices)

    else:
        # embeddings (bs, max_n_tokens, h_dim)
        embeddings = data["bert_embeddings"]
        # mask (bs, max_n_words, max_n_tokens)
        mask = data["bert_alignment"].to(embeddings.device)
        # embeddings (bs, max_n_tokens, h_dim) -> (bs, max_n_words, max_n_tokens, h_dim)_
        embeddings = embeddings.unsqueeze(1).repeat(1, mask.size(1), 1, 1)

        if pooling == "max":
            embeddings.masked_fill_((mask == 0).unsqueeze(-1), -1e30)
            return embeddings.max(2)[0]

        elif pooling == "sum":
            embeddings.masked_fill_((mask == 0).unsqueeze(-1), 0)
            return embeddings.sum(2)

        elif pooling == "avg":
            embeddings.masked_fill_((mask == 0).unsqueeze(-1), 0)
            return embeddings.mean(2)


def init_bert_weights(module, embeddings=True):
    """ Initialize BERT weights"""
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.Embedding) and embeddings:
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class BertEmbedder(nn.Module):
    def __init__(self, bert_model="bert-base-cased", do_lower_case=False, finetune=True, word_pooling="max"):
        super(BertEmbedder, self).__init__()

        # Load pretrained Bert Model
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
        self.bert_encoder = BertModel.from_pretrained(bert_model)
        self.finetune = finetune

        # Word Pooling function to pool subword representations
        assert word_pooling in ["max", "sum", "avg", "first"]
        self.word_pooling = word_pooling

        # Bert embedder properties
        self.w_embed_dim = self.bert_encoder.config.hidden_size
        self.n_heads = self.bert_encoder.config.num_attention_heads
        self.n_layers = self.bert_encoder.config.num_hidden_layers

        self.name = bert_model

    @property
    def device(self):
        return self.bert_encoder.embeddings.word_embeddings.weight.device

    def random_init(self, embeddings=True):
        self.apply(lambda m: init_bert_weights(m, embeddings=embeddings))
        logging.info("WARNING: BERT weights initialized randomly")

    def forward(self, data, keys=["embeddings"]):
        input_ids = data["bert_token_ids"].to(self.device, non_blocking=True)
        attention_mask = mask(data["n_bert_tokens"], device=self.device).float()

        if self.finetune:
            bert_embeddings, _ = self.bert_encoder(input_ids, attention_mask=attention_mask)

        else:
            self.bert_encoder.eval()
            with torch.no_grad():
                bert_embeddings, _ = self.bert_encoder(input_ids, attention_mask=attention_mask)

        data["bert_embeddings"] = bert_embeddings
        data["embeddings"] = token2word_embeddings(data, self.word_pooling)
        data["cls"] = bert_embeddings[:, 0]

        return data
