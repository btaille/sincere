import logging

import torch


def words2ids(words, w2idx, oov=1, cased=True):
    if cased:
        return [w2idx[w] if w in w2idx.keys() else oov for w in words]
    else:
        output = []
        for w in words:
            if w in w2idx.keys():
                output.append(w2idx[w])
            elif w.lower() in w2idx.keys():
                output.append(w2idx[w.lower()])
            else:
                output.append(oov)
        return output


def data_stats(data):
    n_sents = len(data)
    n_tokens = sum([len(d["tokens"]) for d in data])
    n_entities = sum([len(d["entities"]) for d in data])
    n_relations = sum([len(d["relations"]) for d in data])

    logging.info(f"n_sents: {n_sents}")
    logging.info(f"n_tokens: {n_tokens}")
    logging.info(f"n_entities: {n_entities}")
    logging.info(f"n_relations: {n_relations}")


def data_tokens(data):
    return set.union(*[set.union(*[set(sample["tokens"]) for sample in data[split]]) for split in data.keys()])


def pad_1d(x, max_len, padding=0):
    padded = torch.zeros(max_len, device=x.device, dtype=x.dtype).fill_(padding)
    padded[:x.size(0)] = x
    return padded


def pad_2d(x, max_size, padding=0):
    padded = torch.zeros(max_size, device=x.device, dtype=x.dtype).fill_(padding)
    padded[:x.size(0), :x.size(1)] = x
    return padded


def pad_batch_1d(tensors, padding=0):
    max_len = max([t.size(0) for t in tensors])
    padded = [pad_1d(t, max_len, padding=padding) for t in tensors]
    return torch.stack(padded)


def pad_batch_2d(tensors, padding=0):
    max_shape = [len(tensors)] + [max([t.size(d) for t in tensors]) for d in range(len(tensors[0].size()))]
    padded = torch.zeros(max_shape, device=tensors[0].device, dtype=tensors[0].dtype).fill_(padding)

    for i, t in enumerate(tensors):
        padded[i, :t.size(0), :t.size(1)] = t

    return padded


def pad_batch_char_ids(sequences, padding=0, device="cpu", dtype=torch.long):
    max_shape = (len(sequences), max([len(s) for s in sequences]), max([max([len(w) for w in s]) for s in sequences]))
    padded = torch.zeros(max_shape, device=device, dtype=dtype).fill_(padding)

    for i, s in enumerate(sequences):
        for j, w in enumerate(s):
            padded[i, j][:len(w)] = torch.tensor(w)

    return padded


def mask(seqlens, device="cpu"):
    mask = torch.zeros((len(seqlens), max(seqlens)), dtype=torch.bool, device=device)
    for i, l in enumerate(seqlens):
        mask[i, :l] = 1
    return mask
