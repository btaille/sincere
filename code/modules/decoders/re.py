import numpy as np
import torch
from torch import nn

from data.utils import mask


def pair2pred(batch, vocab):
    """Convert span pair predictions into list of relations"""
    pred_entities = batch["pred_entities"]
    batch_size = batch["re_output"].size(0)
    batch_pred_relations = []

    for b in range(batch_size):
        pred_relations = []
        for (head, tail), pred in zip(batch["pair_ids"][b], batch["re_output"][b]):
            for i in pred.nonzero():
                rel = dict()
                rel["head"] = head
                rel["tail"] = tail
                rel["type"] = vocab.relations.idx2val[i.item()]

                if tuple(head) in pred_entities[b] and tuple(tail) in pred_entities[b]:
                    rel["head_type"] = pred_entities[b][tuple(head)] if tuple(head) in pred_entities[b] else "None"
                    rel["tail_type"] = pred_entities[b][tuple(tail)] if tuple(tail) in pred_entities[b] else "None"
                    pred_relations.append(rel)

        batch_pred_relations.append(pred_relations)

    return batch_pred_relations


def get_pair_labels(filtered_pairs, relations=None, entities=None, neg_sampling=0):
    """Return Ground Truth labels along with pairs (with possible negative sampling)"""
    positive_pairs = []
    positive_ids = set()

    if relations is not None and entities is not None:
        for rel in relations:
            head, tail = rel["head"], rel["tail"]

            head_span = (entities[head]["start"], entities[head]["end"])
            tail_span = (entities[tail]["start"], entities[tail]["end"])

            rel_type = rel["type"]

            positive_pairs.append(((head_span, tail_span), rel_type))
            positive_ids.add((head_span, tail_span))

    negative_pairs = [s for s in filtered_pairs if not s in positive_ids]

    if neg_sampling and len(negative_pairs) > neg_sampling:
        negative_pairs = np.array(negative_pairs)
        indices = np.random.choice(np.arange(len(negative_pairs)), size=neg_sampling, replace=False)
        negative_pairs = negative_pairs[indices]

    negative_pairs = [(s, "None") for s in negative_pairs]

    ids, labels = zip(*(positive_pairs + negative_pairs))

    return ids, labels


class REDecoder(nn.Module):
    def __init__(self, entity_dim, vocab, neg_sampling=0, context_dim=0, biaffine=False, dropout=0.0,
                 pooling_fn="max", chunk_size=1000):
        """Linear Scorer : MLP([ent1, ent2])   +  optionally [pooled_middle_context, bilinear(ent1, ent2)]"""
        super().__init__()
        self.entity_dim = entity_dim
        self.input_dim = entity_dim * 2
        self.vocab = vocab
        self.neg_sampling = neg_sampling
        self.chunk_size = chunk_size

        self.context_dim = context_dim
        # Add context representation of same size than both entities
        if self.context_dim:
            self.context_pool = lambda x: x.max(0)[0]
            self.input_dim += self.context_dim

        if pooling_fn == "max":
            self.pool = lambda x: x.max(0)[0]
        else:
            raise NotImplementedError

        self.biaffine = biaffine
        # Add bilinear term
        if self.biaffine:
            self.bilinear = nn.Bilinear(self.entity_dim, self.entity_dim, len(self.vocab.relations.idx2val), bias=False)

        self.dropout = dropout
        self.drop = nn.Dropout(dropout)

        self.linear = nn.Linear(self.input_dim, len(self.vocab.relations.idx2val))

        self.supervision = "relations"
        self.name = "re"

    def forward(self, data, input_key="encoded"):
        batch_size, max_n_words, context_dim = data[input_key].size()

        if self.context_dim:
            assert context_dim == self.context_dim

        device = data[input_key].device

        all_pair_ids = []
        all_labels = []
        all_filtered_span_ids = []

        for b in range(batch_size):

            # Filter entity spans AND SAMPLE
            gt_spans = [(ent["start"], ent["end"]) for ent in data["entities"][b]]
            pred_spans = data["pred_entities"][b].keys()

            if self.training:
                # Keep GT spans and Predicted spans in training
                filtered_spans = list(set(gt_spans).union(set(pred_spans)))
            else:
                # Keep only Predicted spans in inference
                filtered_spans = pred_spans

            all_filtered_span_ids.append(filtered_spans)

            # Needs at least 2 spans to classify relations
            if len(filtered_spans) > 1:
                # Get all span pairs and labels
                filtered_pairs = [(a, b) for a in filtered_spans for b in filtered_spans if not a == b]

                # If training, possible negative sampling
                if self.training:
                    filtered_pairs, filtered_pairs_labels = get_pair_labels(filtered_pairs,
                                                                            data["relations"][b],
                                                                            data["entities"][b],
                                                                            neg_sampling=self.neg_sampling)
                # Else do not pass ground_truth information
                # /!\ This results in an inaccurate loss computation in eval
                # But it enables not to consider GT entities not predicted as entities
                else:
                    filtered_pairs, filtered_pairs_labels = get_pair_labels(filtered_pairs,
                                                                            neg_sampling=False)


            # Otherwise dummy relation
            else:
                filtered_pairs = [((0, 1), (0, 1))]
                filtered_pairs_labels = ["None"]

            all_pair_ids.append(filtered_pairs)
            all_labels.append(filtered_pairs_labels)

        data["pair_ids"] = all_pair_ids
        data["n_pairs"] = [len(p) for p in all_pair_ids]
        max_n_relations = max(data["n_pairs"])

        # If IOBES NER : compute span representations (already done in Span NER)
        if not "span_ids" in data.keys():
            n_spans = [len(s) for s in all_filtered_span_ids]
            max_n_spans = max(n_spans)
            span_representations = torch.zeros((batch_size, max_n_spans, context_dim), device=device)

            for b in range(batch_size):
                for i, (start, end) in enumerate(all_filtered_span_ids[b]):
                    span_representations[b, i] = self.pool(data[input_key][b, start: end, :])

            data["span_pooled"] = span_representations
            data["n_spans"] = n_spans
            data["span_ids"] = all_filtered_span_ids

        # Get span pairs representations and labels
        all_pair_representations = torch.zeros((batch_size, max_n_relations, self.input_dim), device=device)
        targets = torch.zeros((batch_size, max_n_relations, len(self.vocab.relations.val2idx)), device=device)

        for b in range(batch_size):
            # Needs at least 2 spans to classify relations
            if len(all_filtered_span_ids[b]) > 1:
                # Dict that maps a span (start, end) to its index in span_ids to retrieve pooled representations
                span2idx = {span: idx for idx, span in enumerate(data["span_ids"][b])}

                filtered_pairs_representations = []
                filtered_pairs_context = []

                for i, (arg1, arg2) in enumerate(all_pair_ids[b]):
                    # Concat both argument representations for each relation

                    filtered_pairs_representations.append(
                        torch.cat([data["span_pooled"][b][span2idx[tuple(arg1)]],
                                   data["span_pooled"][b][span2idx[tuple(arg2)]]], -1))

                    # Get pooled Middle context
                    if self.context_dim:
                        if tuple(arg1) < tuple(arg2):
                            begin, end = arg1[1], arg2[0]
                        else:
                            begin, end = arg2[1], arg1[0]

                        if end - begin > 0:
                            filtered_pairs_context.append(self.pool(data[input_key][b, begin:end, :]))
                        else:
                            filtered_pairs_context.append(torch.zeros(context_dim, device=device))

                    # Add label to targets if not None
                    if all_labels[b][i] != "None":
                        targets[b, i, self.vocab.relations.val2idx[all_labels[b][i]]] = 1

                filtered_pairs_representations = torch.stack(filtered_pairs_representations)

                if self.context_dim:
                    filtered_pairs_context = torch.stack(filtered_pairs_context)
                    filtered_pairs_representations = torch.cat([filtered_pairs_representations, filtered_pairs_context],
                                                               -1)
                all_pair_representations[b, :len(filtered_pairs_representations), :] = filtered_pairs_representations

        data["pair_pooled"] = all_pair_representations
        data["pair_tags"] = targets

        # Classify pairs
        scores = torch.zeros((batch_size, max_n_relations, len(self.vocab.relations.val2idx)), device=device)
        for i in range(0, max_n_relations, self.chunk_size):
            pair_chunk_rep = data["pair_pooled"][:, i:i + self.chunk_size]
            scores[:, i:i + self.chunk_size] = self.linear(self.drop(pair_chunk_rep))

            if self.biaffine:
                head = pair_chunk_rep[:, :, :self.entity_dim].contiguous()
                tail = pair_chunk_rep[:, :, self.entity_dim: 2 * self.entity_dim].contiguous()

                scores[:, i:i + self.chunk_size] += self.bilinear(self.drop(head), self.drop(tail))

        data[f"{self.name}_scores"] = nn.Sigmoid()(scores)
        data[f"{self.name}_output"] = data[f"{self.name}_scores"] > 0.5

        # Convert pair predictions into list of relations
        data["pred_relations"] = pair2pred(data, self.vocab)

    def loss(self, data):
        scores = data[f"{self.name}_scores"]
        device = scores.device
        tags = data["pair_tags"]

        seqlens = data["n_pairs"]
        loss_mask = mask(seqlens, device=device)

        # loss = nn.BCELoss(reduction="none")(scores.view(-1, len(self.vocab.relations.val2idx)), tags.view(-1))
        loss = nn.BCELoss(reduction="none")(scores, tags)

        masked_loss = loss * loss_mask.float().unsqueeze(-1)

        # normalize per tag
        return masked_loss.sum() / loss_mask.sum()
