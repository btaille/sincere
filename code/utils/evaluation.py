import logging
import torch
import numpy as np
from tqdm import tqdm


def evaluate_ner(model, data_loader):
    losses = []
    pred_entities = []
    gt_entities = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            model(batch, task="ner")
            gt_entities.extend(batch["entities"])
            pred_entities.extend(
                [[{"start": span[0], "end": span[1], "type": ent_type} for span, ent_type in s.items()] for s in
                 batch["pred_entities"]])
            losses.append(model.decoder.loss(batch, "ner").item())

    return pred_entities, np.mean(losses), ner_score(pred_entities, gt_entities, model.decoder.decoders["ner"].vocab)


def evaluate_re(model, data_loader, mode="strict"):
    losses = []
    pred_relations = []
    gt_relations = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            model(batch, task="re")

            pred_relations.extend(batch["pred_relations"])

            losses.append(model.decoder.loss(batch, "re").item())

            for b in range(len(batch["relations"])):
                rel_sent = []
                for rel in batch["relations"][b]:
                    rel["head_id"] = rel["head"]
                    head = batch["entities"][b][rel["head_id"]]
                    rel["head"] = (head["start"], head["end"])
                    rel["head_type"] = head["type"]

                    rel["tail_id"] = rel["tail"]
                    tail = batch["entities"][b][rel["tail_id"]]
                    rel["tail"] = (tail["start"], tail["end"])
                    rel["tail_type"] = tail["type"]

                    rel_sent.append(rel)

                gt_relations.append(rel_sent)

    return pred_relations, np.mean(losses), re_score(pred_relations, gt_relations, model.decoder.decoders["re"].vocab,
                                                     mode=mode)


def ner_score(pred_entities, gt_entities, vocab):
    """Evaluate NER predictions

    Args:
        pred_entities (list) :  list of list of predicted entities (several entities in each sentence)
        gt_entities (list) :    list of list of ground truth entities

            entity = {"start": start_idx (inclusive),
                      "end": end_idx (exclusive),
                      "type": ent_type}

        vocab (Vocab) :         dataset vocabulary"""
    assert len(pred_entities) == len(gt_entities)
    entity_types = [v for v in vocab.entities.idx2val.values() if not v == "None"]

    scores = {ent: {"tp": 0, "fp": 0, "fn": 0} for ent in entity_types + ["ALL"]}

    # Count GT entities and Predicted entities
    n_sents = len(gt_entities)
    n_phrases = sum([len([ent for ent in sent]) for sent in gt_entities])
    n_found = sum([len([ent for ent in sent]) for sent in pred_entities])

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(pred_entities, gt_entities):
        for ent_type in entity_types:
            pred_ents = {(ent["start"], ent["end"]) for ent in pred_sent if ent["type"] == ent_type}
            gt_ents = {(ent["start"], ent["end"]) for ent in gt_sent if ent["type"] == ent_type}

            scores[ent_type]["tp"] += len(pred_ents & gt_ents)
            scores[ent_type]["fp"] += len(pred_ents - gt_ents)
            scores[ent_type]["fn"] += len(gt_ents - pred_ents)

    # Compute per entity Precision / Recall / F1
    for ent_type in scores.keys():
        if scores[ent_type]["tp"]:
            scores[ent_type]["p"] = 100 * scores[ent_type]["tp"] / (scores[ent_type]["fp"] + scores[ent_type]["tp"])
            scores[ent_type]["r"] = 100 * scores[ent_type]["tp"] / (scores[ent_type]["fn"] + scores[ent_type]["tp"])
        else:
            scores[ent_type]["p"], scores[ent_type]["r"] = 0, 0

        if not scores[ent_type]["p"] + scores[ent_type]["r"] == 0:
            scores[ent_type]["f1"] = 2 * scores[ent_type]["p"] * scores[ent_type]["r"] / (
                    scores[ent_type]["p"] + scores[ent_type]["r"])
        else:
            scores[ent_type]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[ent_type]["tp"] for ent_type in entity_types])
    fp = sum([scores[ent_type]["fp"] for ent_type in entity_types])
    fn = sum([scores[ent_type]["fn"] for ent_type in entity_types])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in entity_types])
    scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in entity_types])
    scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in entity_types])

    logging.info(
        "processed {} sentences with {} phrases; found: {} phrases; correct: {}.".format(n_sents, n_phrases, n_found,
                                                                                         tp))
    logging.info(
        "\tALL\t TP: {};\tFP: {};\tFN: {}".format(
            scores["ALL"]["tp"],
            scores["ALL"]["fp"],
            scores["ALL"]["fn"]))
    logging.info(
        "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(
            precision,
            recall,
            f1))
    logging.info(
        "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
            scores["ALL"]["Macro_p"],
            scores["ALL"]["Macro_r"],
            scores["ALL"]["Macro_f1"]))

    for ent_type in entity_types:
        logging.info("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
            ent_type,
            scores[ent_type]["tp"],
            scores[ent_type]["fp"],
            scores[ent_type]["fn"],
            scores[ent_type]["p"],
            scores[ent_type]["r"],
            scores[ent_type]["f1"],
            scores[ent_type]["tp"] +
            scores[ent_type][
                "fp"]))

    return scores


def re_score(pred_relations, gt_relations, vocab, mode="strict"):
    """Evaluate RE predictions

    Args:
        pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
        gt_relations (list) :    list of list of ground truth relations

            rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
                    "tail": (start_idx (inclusive), end_idx (exclusive)),
                    "head_type": ent_type,
                    "tail_type": ent_type,
                    "type": rel_type}

        vocab (Vocab) :         dataset vocabulary
        mode (str) :            in 'strict' or 'boundaries' """

    assert mode in ["strict", "boundaries"]

    relation_types = [v for v in vocab.relations.idx2val.values() if not v == "None"]
    scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in relation_types + ["ALL"]}

    # Count GT relations and Predicted relations
    n_sents = len(gt_relations)
    n_rels = sum([len([rel for rel in sent]) for sent in gt_relations])
    n_found = sum([len([rel for rel in sent]) for sent in pred_relations])

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(pred_relations, gt_relations):
        for rel_type in relation_types:
            # strict mode takes argument types into account
            if mode == "strict":
                pred_rels = {(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in pred_sent if
                             rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in gt_sent if
                           rel["type"] == rel_type}

            # boundaries mode only takes argument spans into account
            elif mode == "boundaries":
                pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent if rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["tail"]) for rel in gt_sent if rel["type"] == rel_type}

            scores[rel_type]["tp"] += len(pred_rels & gt_rels)
            scores[rel_type]["fp"] += len(pred_rels - gt_rels)
            scores[rel_type]["fn"] += len(gt_rels - pred_rels)

    # Compute per entity Precision / Recall / F1
    for rel_type in scores.keys():
        if scores[rel_type]["tp"]:
            scores[rel_type]["p"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fp"] + scores[rel_type]["tp"])
            scores[rel_type]["r"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fn"] + scores[rel_type]["tp"])
        else:
            scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0

        if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
            scores[rel_type]["f1"] = 2 * scores[rel_type]["p"] * scores[rel_type]["r"] / (
                    scores[rel_type]["p"] + scores[rel_type]["r"])
        else:
            scores[rel_type]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[rel_type]["tp"] for rel_type in relation_types])
    fp = sum([scores[rel_type]["fp"] for rel_type in relation_types])
    fn = sum([scores[rel_type]["fn"] for rel_type in relation_types])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in relation_types])
    scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in relation_types])
    scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in relation_types])

    logging.info(f"RE Evaluation in *** {mode.upper()} *** mode")

    logging.info(
        "processed {} sentences with {} relations; found: {} relations; correct: {}.".format(n_sents, n_rels, n_found,
                                                                                             tp))
    logging.info(
        "\tALL\t TP: {};\tFP: {};\tFN: {}".format(
            scores["ALL"]["tp"],
            scores["ALL"]["fp"],
            scores["ALL"]["fn"]))
    logging.info(
        "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(
            precision,
            recall,
            f1))
    logging.info(
        "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
            scores["ALL"]["Macro_p"],
            scores["ALL"]["Macro_r"],
            scores["ALL"]["Macro_f1"]))

    for rel_type in relation_types:
        logging.info("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
            rel_type,
            scores[rel_type]["tp"],
            scores[rel_type]["fp"],
            scores[rel_type]["fn"],
            scores[rel_type]["p"],
            scores[rel_type]["r"],
            scores[rel_type]["f1"],
            scores[rel_type]["tp"] +
            scores[rel_type][
                "fp"]))

    return scores
