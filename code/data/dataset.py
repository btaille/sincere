import copy
import json
import logging

import numpy as np
import torch
from transformers import BertTokenizer

from data.utils import data_stats
from data.utils import words2ids, pad_batch_1d, data_tokens, pad_2d, pad_1d
from modules.embedders.bert import bert_tokenize


class Index:
    def __init__(self, values, special_values=None):
        values = sorted(values)

        if special_values is not None:
            values = special_values + values

        idx2val = {i: v for i, v in enumerate(values)}

        self.idx2val = idx2val
        self.val2idx = {v: k for k, v in idx2val.items()}


class Vocab:
    def __init__(self, entities=None, relations=None, words=None):
        self.words, self.entities, self.relations, self.chars = None, None, None, None

        if words is not None:
            self.words = Index(words, special_values=["<PAD>", "<UNK>"])
        if entities is not None:
            self.entities = Index(entities, special_values=["None"])
        if relations is not None:
            self.relations = Index(relations)

        self.iobes = None

    def compute_char_index(self):
        assert self.words is not None
        words = list(self.words.idx2val.values())
        chars = list(set("".join(words)))
        self.chars = Index(chars, special_values=["<PAD>", "<UNK>"])

    def compute_iobes_index(self):
        assert self.entities is not None
        iobes = []
        for ent_type in [t for t in self.entities.idx2val.values() if not t == "None"]:
            iobes.extend([f"{prefix}-{ent_type}" for prefix in "BIES"])

        self.iobes = Index(iobes, special_values=["O"])


def data_vocab(data):
    entities = set()
    relations = set()

    tokens = data_tokens(data)

    for split in data.keys():
        for sent in data[split]:
            entities = entities.union({e["type"] for e in sent["entities"]})
            relations = relations.union({r["type"] for r in sent["relations"]})

    vocab = Vocab(entities=entities, relations=relations, words=tokens)
    vocab.compute_char_index()
    vocab.compute_iobes_index()

    return vocab


class Dataset(object):
    def __init__(self, data, vocab):
        super().__init__()
        self.vocab = vocab
        self.data = data

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def prepare_dataset(self, config=None, bert_model=None, do_lower_case=False, device="cpu"):
        bert_tokenizer = None
        if config is not None:
            device = config.device
            for emb in config.embedder:
                if "bert" in emb:
                    bert_tokenizer = BertTokenizer.from_pretrained(f"{emb}-cased", do_lower_case=do_lower_case)
                    break
        elif bert_model is not None:
            bert_tokenizer = BertTokenizer.from_pretrained(f"{bert_model}-cased", do_lower_case=do_lower_case)

        for d in self.data:
            d["words"] = d["tokens"]
            d["n_words"] = len(d["words"])
            d["chars"] = [[c for c in w] for w in d["words"]]
            d["n_chars"] = [len(w) for w in d["chars"]]

            if self.vocab.words is not None:
                d["word_ids"] = torch.tensor(words2ids(d["words"], self.vocab.words.val2idx)).to(device)

            if self.vocab.chars is not None:
                #
                d["char_ids"] = pad_batch_1d(
                    [torch.tensor(words2ids(w, self.vocab.chars.val2idx)).to(device) for w in d["chars"]])

            if self.vocab.iobes is not None:
                iobes = ["O"] * d["n_words"]

                for ent in d["entities"]:
                    if ent["end"] - ent["start"] == 1:
                        iobes[ent["start"]] = f"S-{ent['type']}"

                    else:
                        iobes[ent["start"]] = f"B-{ent['type']}"
                        iobes[ent["end"] - 1] = f"E-{ent['type']}"

                        iobes[ent["start"] + 1: ent["end"] - 1] = [f"I-{ent['type']}"] * (ent["end"] - ent["start"] - 2)

                d["iobes"] = iobes
                d["iobes_ids"] = torch.tensor(words2ids(iobes, self.vocab.iobes.val2idx)).to(device)

            if bert_tokenizer is not None:
                d.update(bert_tokenize(d, bert_tokenizer, device=device))

        self.pad_dataset()

    def pad_dataset(self):
        max_n_words = max([d["n_words"] for d in self.data])
        max_n_chars = max([max(d["n_chars"]) for d in self.data])

        bert = False

        if "n_bert_tokens" in self.data[0].keys():
            max_n_bert_tokens = max([d["n_bert_tokens"] for d in self.data])
            bert = True

        for d in self.data:
            d["word_ids"] = pad_1d(d["word_ids"], max_n_words)
            d["char_ids"] = pad_2d(d["char_ids"], (max_n_words, max_n_chars))

            if "iobes_ids" in d.keys():
                d["iobes_ids"] = pad_1d(d["iobes_ids"], max_n_words)

            if bert:
                d["bert_token_ids"] = pad_1d(d["bert_token_ids"], max_n_bert_tokens)
                d["bert_indices"] = pad_1d(d["bert_indices"], max_n_words)
                d["bert_alignment"] = pad_2d(d["bert_alignment"], (max_n_words, max_n_bert_tokens))


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    max_words = max([s["n_words"] for s in batch])
    max_chars = max([max(s["n_chars"]) for s in batch])

    if "n_bert_tokens" in batch[0].keys():
        max_tokens = max([s["n_bert_tokens"] for s in batch])

    for k in keys:
        if k in ["words", "chars", "sent", "entities", "relations", "n_words", "iobes", "bert_tokens", "n_bert_tokens"]:
            padded_batch[k] = [s[k] for s in batch]

        elif k in ["word_ids", "iobes_ids", "bert_indices"]:
            padded_batch[k] = torch.stack([s[k][:max_words] for s in batch])

        elif k in ["bert_token_ids"]:
            padded_batch[k] = torch.stack([s[k][:max_tokens] for s in batch])

        elif k in ["char_ids"]:
            padded_batch[k] = torch.stack([s[k][:max_words, :max_chars] for s in batch])

        elif k in ["bert_alignment"]:
            padded_batch[k] = torch.stack([s[k][:max_words, :max_tokens] for s in batch])

        # Pad n_chars with 1 for each <PAD> word token
        elif k == "n_chars":
            padded_batch["n_chars"] = [s["n_chars"] + [1] * (max_words - len(s["n_chars"])) for s in batch]

    return padded_batch


class DataIterator(object):
    def __init__(self, dataset, device="cpu", batch_size=1, drop_last=False, shuffle=False,
                 collate_fn=collate_fn_padding):
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.device = device
        self.drop_last = drop_last

        self.dataset = dataset

        self.i = 0
        self.max = len(self.dataset)

        if drop_last:
            self.max -= self.max % self.batch_size

        self.n_batches = self.max // self.batch_size + int(self.max % self.batch_size > 0)
        self.indices = np.arange(self.max)

        if self.shuffle:
            self.shuffle_data()

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.max:
            range = (self.i, min(self.i + self.batch_size, self.max))
            self.i += self.batch_size

            return self.collate_fn([copy.deepcopy(self.dataset[i]) for i in self.indices[range[0]: range[1]]])
        raise StopIteration

    def shuffle_data(self):
        self.indices = np.random.permutation(self.indices)

    def reinit(self):
        self.i = 0
        if self.shuffle:
            self.shuffle_data()


def init_data_iterators(datasets, batch_size=1, shuffle_train=True, drop_last=False, collate_fn=collate_fn_padding,
                        train_key="train"):
    data_iterators = {
        split: DataIterator(datasets[split], batch_size=batch_size, shuffle=shuffle_train, drop_last=drop_last,
                            collate_fn=collate_fn) if split == train_key
        else DataIterator(datasets[split], batch_size=batch_size, shuffle=False, drop_last=drop_last,
                          collate_fn=collate_fn)
        for split in datasets.keys()}

    return data_iterators


def dump(data, output_path):
    with open(output_path, "w") as file:
        json.dump(data, file)


def load_data(input_path, verbose=True):
    with open(input_path, "r") as file:
        data = json.load(file)

    if verbose:
        for split in data.keys():
            logging.info(f"\n{split}")
            data_stats(data[split])

    return data, data_vocab(data)
