import numpy as np
import torch
from torch import nn

from data.utils import mask, words2ids


def get_all_span_ids(sent_len, max_span_len=10):
    "Return all spans (start (inclusive), end (exclusive)) with length <= max_span_len"
    span_indices = set()
    for i in range(sent_len):
        for k in range(1, max_span_len + 1):
            # (start indice (inclusive), end_indice (exclusive)) in original sentence :
            span_indices.add((i, min(i + k, sent_len)))

    return span_indices


def get_span_labels(spans, entities=None, neg_sampling=0):
    "Return Ground Truth labels along with spans (with possible negative sampling)"
    positive_spans = []
    positive_ids = set()

    if entities is not None:
        for ent in entities:
            positive_spans.append(((ent["start"], ent["end"]), ent["type"]))
            positive_ids.add((ent["start"], ent["end"]))

    negative_spans = [s for s in spans if s not in positive_ids]

    if neg_sampling and len(negative_spans) > neg_sampling:
        negative_spans = np.array(negative_spans)
        indices = np.random.choice(np.arange(len(negative_spans)), size=neg_sampling, replace=False)
        negative_spans = negative_spans[indices]

    negative_spans = [((s[0], s[1]), "None") for s in negative_spans]

    return zip(*(positive_spans + negative_spans))


def span2pred(batch, vocab):
    "Convert span predictions into list of entities ({'start', 'end', 'type'})"
    batch_size = batch["ner_output"].size(0)
    batch_pred_entities = []

    for b in range(batch_size):
        pred_entities = {}
        for span, pred in zip(batch["span_ids"][b], batch["ner_output"][b]):
            if not pred.item() == vocab.entities.val2idx["None"]:
                # ent = {span[0], "end": span[1], "type": vocab.entities.idx2val[pred.item()]}
                pred_entities[(span[0], span[1])] = vocab.entities.idx2val[pred.item()]

        batch_pred_entities.append(pred_entities)

    return batch_pred_entities


def iobes2iob(iobes):
    "Converts a list of IOBES tags to IOB scheme."
    convert_dict = {pfx: pfx for pfx in "IOB"}
    convert_dict.update({"S": "B", "E": "I"})
    return [convert_dict[t[0]] + t[1:] if not t == "O" else "O" for t in iobes]


def extract_iob(iob_tags):
    "Convert list of IOB tags into list of entities ({'start', 'end', 'type'})"
    entities = {}

    tmp_indices = None
    tmp_type = "O"
    for i, t in enumerate(iob_tags):
        if t[0] == "B":
            if tmp_indices is not None:
                entities[(tmp_indices[0], tmp_indices[-1] + 1)] = tmp_type
                # entities.append({"start": tmp_indices[0], "end": tmp_indices[-1] + 1, "type": tmp_type})
            tmp_type = "-".join(t.split("-")[1:])
            tmp_indices = [i]

        elif t[0] == "O":
            if tmp_indices is not None:
                entities[(tmp_indices[0], tmp_indices[-1] + 1)] = tmp_type
                # entities.append({"start": tmp_indices[0], "end": tmp_indices[-1] + 1, "type": tmp_type})
            tmp_type = None
            tmp_indices = None

        elif t[0] == "I":
            if "-".join(t.split("-")[1:]) == tmp_type and i == tmp_indices[-1] + 1:
                tmp_indices += [i]
            else:
                if tmp_indices is not None:
                    entities[(tmp_indices[0], tmp_indices[-1] + 1)] = tmp_type
                    # entities.append({"start": tmp_indices[0], "end": tmp_indices[-1] + 1, "type": tmp_type})
                tmp_type = "-".join(t.split("-")[1:])
                tmp_indices = [i]

    if tmp_indices is not None:
        entities[(tmp_indices[0], tmp_indices[-1] + 1)] = tmp_type
        # entities.append({"start": tmp_indices[0], "end": tmp_indices[-1] + 1, "type": tmp_type})

    return entities


def iobes2pred(batch, vocab):
    "Convert iobes predictions into list of entities ({'start', 'end', 'type'})"
    pred_iobes_ids = batch["ner_output"]
    pred_iobes = [words2ids(s.cpu().numpy(), vocab.iobes.idx2val) for s in pred_iobes_ids]
    pred_iobes = [p[:n_words] for p, n_words in zip(pred_iobes, batch["n_words"])]

    return [extract_iob(iobes2iob(p)) for p in pred_iobes]


class SpanNERDecoder(nn.Module):
    def __init__(self, input_dim, vocab, neg_sampling=0, max_span_len=10, span_len_embedding_dim=25, pooling_fn="max",
                 dropout=0., chunk_size=1000, use_cls=False):
        super().__init__()
        self.vocab = vocab
        self.neg_sampling = neg_sampling
        self.chunk_size = chunk_size

        if pooling_fn == "max":
            self.pool = lambda x: x.max(0)[0]
        else:
            raise NotImplementedError

        self.input_dim = input_dim
        self.max_span_len = max_span_len

        # Use CLS
        self.use_cls = use_cls
        if self.use_cls:
            self.input_dim *= 2

        # Span length embeddings
        self.span_len_embedding_dim = span_len_embedding_dim
        if self.span_len_embedding_dim:
            self.span_len_embedder = nn.Embedding(max_span_len, self.span_len_embedding_dim)
            self.input_dim += self.span_len_embedding_dim

        # Linear Layer
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(self.input_dim, len(self.vocab.entities.idx2val))

        # Decoder name and tagging scheme
        self.name = "ner"
        self.scheme = "span"
        self.supervision = "span_tags"

    def forward(self, data, input_key="encoded", cls_key="cls"):
        word_encodings = data[input_key]
        batch_size, _, encoding_dim = word_encodings.size()
        device = word_encodings.device

        # Get all spans and their corresponding labels
        all_span_ids = []
        all_labels = []

        for b in range(batch_size):
            span_ids = get_all_span_ids(data["n_words"][b], self.max_span_len)
            # If training, possible negative sampling
            if self.training:
                span_ids, labels = get_span_labels(span_ids, data["entities"][b], neg_sampling=self.neg_sampling)
            # Else no negative sampling (GT is only used to compute loss)
            else:
                span_ids, labels = get_span_labels(span_ids, data["entities"][b] if "entities" in data.keys() else None,
                                                   neg_sampling=False)

            all_span_ids.append(span_ids)
            all_labels.append(labels)

        n_spans = [len(spans) for spans in all_span_ids]
        max_n_spans = max(n_spans)

        # Pool word representations in span_representations
        span_representations = torch.zeros((batch_size, max_n_spans, self.input_dim), device=device)
        targets = torch.zeros((batch_size, max_n_spans), device=device, dtype=torch.long)

        for b in range(batch_size):
            for i, ((start, end), label) in enumerate(zip(all_span_ids[b], all_labels[b])):
                if not end - start > self.max_span_len:
                    targets[b, i] = self.vocab.entities.val2idx[label]

                    span_representation = self.pool(word_encodings[b, start: end, :])

                    # Concat span length embedding
                    if self.span_len_embedding_dim:
                        span_len = torch.tensor(end - start - 1, dtype=torch.long, device=device)
                        span_representation = torch.cat([span_representation, self.span_len_embedder(span_len)], -1)

                    # Concat CLS
                    if self.use_cls:
                        span_representation = torch.cat([span_representation, data[cls_key]], -1)

                    span_representations[b, i] = span_representation

        data["span_pooled"] = span_representations
        data["n_spans"] = n_spans
        data["span_ids"] = all_span_ids
        data["span_tags"] = targets

        # Classify spans in entities
        scores = torch.zeros((batch_size, max_n_spans, len(self.vocab.entities.val2idx)), device=device)

        for i in range(0, max_n_spans, self.chunk_size):
            scores[:, i:i + self.chunk_size] = self.linear(self.drop(data["span_pooled"][:, i:i + self.chunk_size]))

        data[f"{self.name}_scores"] = scores
        data[f"{self.name}_output"] = torch.argmax(scores, dim=-1)

        # Convert span predictions into list of entities
        data["pred_entities"] = span2pred(data, self.vocab)

    def loss(self, data):
        scores = data[f"{self.name}_scores"]
        device = scores.device
        tags = data["span_tags"].long()

        seqlens = data["n_spans"]
        loss_mask = mask(seqlens, device=device)

        loss = nn.CrossEntropyLoss(reduction="none")(scores.view(-1, len(self.vocab.entities.val2idx)), tags.view(-1))
        masked_loss = loss * loss_mask.view(-1).float()

        # normalize per tag
        return masked_loss.sum() / loss_mask.sum()


class IobesNERDecoder(nn.Module):
    def __init__(self, input_dim, vocab, dropout=0.):
        super().__init__()
        self.vocab = vocab

        self.input_dim = input_dim

        # Linear Layer
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(self.input_dim, len(self.vocab.iobes.idx2val))

        # Decoder name and tagging scheme
        self.name = "ner"
        self.scheme = "iobes"
        self.supervision = "iobes_ids"

    def forward(self, data, input_key="encoded"):
        word_encodings = data[input_key]

        # Classify spans in entities
        scores = self.linear(self.drop(word_encodings))

        data[f"{self.name}_scores"] = scores
        data[f"{self.name}_output"] = torch.argmax(scores, dim=-1)

        # Convert iobes predictions into list of entities
        data["pred_entities"] = iobes2pred(data, self.vocab)

    def loss(self, data):
        scores = data[f"{self.name}_scores"]
        device = scores.device
        tags = data[self.supervision].to(device, non_blocking=True)

        seqlens = data["n_words"]
        loss_mask = mask(seqlens, device=device)

        loss = nn.CrossEntropyLoss(reduction="none")(scores.view(-1, len(self.vocab.iobes.val2idx)), tags.view(-1))
        masked_loss = loss * loss_mask.view(-1).float()

        # normalize per tag
        return masked_loss.sum() / loss_mask.sum()
