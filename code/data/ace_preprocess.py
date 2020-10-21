import logging
import re
from glob import glob

from data.utils import data_stats
from data.dataset import data_vocab
import numpy as np


def char2token_offsets(start, sents, word):
    # Convert start and end offsets from number of characters to number of tokens
    sentlens = [len(s) for s in sents]
    sid = 0
    offset = 0
    for l in sentlens:
        if offset + l < start:
            offset += l + 1
            sid += 1
        else:
            break

    sent = sents[sid]
    white_spaces = re.findall('\s+', sent)
    space_lens = []

    for i in range(0, len(white_spaces)):
        space_lens.append(len(white_spaces[i]))

    wordlens = [len(w) for w in sent.split()]

    if len(sent) - len(sent.lstrip()):
        wordlens[0] += space_lens[0]
        space_lens = space_lens[1:]

    woffset = 0
    wid = 0
    beg_id = None

    for wl, sl in zip(wordlens, space_lens):
        if woffset + wl < start - offset:
            woffset += wl + sl
            wid += 1
        else:
            beg_id = wid
            break

    if beg_id is None:
        beg_id = len(sent.split()) - 1

    try:
        assert sent.split()[beg_id] == word[0] or re.sub("&amp;|&AMP;", "&", sent.split()[beg_id]) == word[0]
    except:
        print("sentlen", len(sent))
        print("start - offset", start - offset)
        print("sent[start-offset:]", sent[start - offset:])
        print(wordlens)
        print(space_lens)

        print("Expected", word)
        print("Found", sent.split()[beg_id])

    end_id = beg_id + len(word)

    return sid, beg_id, end_id


def read_doc_annotation(txt_path, ann_path):
    docid = txt_path.split("/")[-1].split(".txt")[0]

    # Split text into sentences
    with open(txt_path, "r", encoding="utf8") as file:
        text = file.read()
        sents = text.split("\n")

    # Transform textual annotations into:
    # 1. document-level ent and rel dictionaries
    entities = {}
    relations = {}
    # 2. sentence-level ent and rel dictionaries
    data = [{"tokens": sents[sid].split(),
             "entities": [],
             "relations": []}
            for sid in range(len(sents))]

    # Read annotation file
    with open(ann_path, "r", encoding="utf8") as file:
        text = file.read()
        annotations = [l for l in text.split("\n") if len(l.strip())]

    for annotation in annotations:
        # Entity annotation
        if annotation[0] == "T":
            eid, etype, start, end = annotation.split()[:4]
            word = annotation.split()[4:]

            # Convert start and end offsets from character id to token id
            sid, start_id, end_id = char2token_offsets(int(start), sents, word)

            # Add to document-level
            entities[eid] = {"sent_id": sid,
                             "start": start_id,
                             "end": end_id,
                             "type": etype,
                             "new_ent_id": len(data[sid]["entities"]),
                             "word": word}

            # Add to sentence-level
            data[sid]["entities"].append({"start": start_id,
                                          "end": end_id,
                                          "type": etype,
                                          "original_id": eid})

        # Relation annotation
        elif annotation[0] == "R":
            rid, rtype, eid1, eid2 = annotation.split()

            ent1 = entities[eid1.split(":")[1]]
            ent2 = entities[eid2.split(":")[1]]

            sid1, start_id1, end_id1, etype1 = ent1["sent_id"], ent1["start"], ent1["end"], ent1["type"]
            sid2, start_id2, end_id2, etype2 = ent2["sent_id"], ent2["start"], ent2["end"], ent2["type"]

            # Only consider intra-sentence relations between different entities
            if sid1 == sid2 and not start_id1 == start_id2:

                # Add to document-level
                relations[rid] = {"type": rtype,
                                  "head": eid1.split(":")[1],
                                  "tail": eid2.split(":")[1]}

                # Add to sentence-level

                data[sid1]["relations"].append({"type": rtype,
                                                "head": ent1["new_ent_id"],
                                                "tail": ent2["new_ent_id"],
                                                "original_id": rid})

            elif not start_id1 == start_id2:
                logging.info("Dumped 1 relation between different sentences in {}".format(txt_path))
            else:
                logging.info("Dumped 1 relation involving the same argument twice")

        else:
            assert len(annotation) == 0

    return data, entities, relations


def load_ace05(data_path, discard_first_lines=0):
    """  Load ACE05 data formatted using (Miwa and Bansal 2016)'s code: https://github.com/tticoin/LSTM-ER
     The first 3 lines correspond to metadata and could be dropped
    """
    data = {"train": [],
            "dev": [],
            "test": []}

    for split in ["train", "dev", "test"]:

        # paths = [path for path in glob(data_path + "corpus/{}/*.txt".format(split)) if not "split.txt" in path]
        paths = glob(data_path + "corpus/{}/*.split.txt".format(split))

        logging.info("\n{}: {} documents".format(split, len(paths)))
        for txt_path in paths:
            ann_path = re.sub(".txt", ".ann", txt_path)

            doc_data, doc_entities, doc_relations = read_doc_annotation(txt_path, ann_path)

            for d in doc_data[discard_first_lines:]:
                # Discard empty sentences
                if len(d["tokens"]):
                    data[split].append(d)

    for split in ["train", "dev", "test"]:
        logging.info("\n" + split)
        data_stats(data[split])

    return data, data_vocab(data)


def load_ace04(data_path, discard_first_lines=0):
    """  Load ACE04 data formatted using (Miwa and Bansal 2016)'s code: https://github.com/tticoin/LSTM-ER
    The first 3 lines correspond to metadata and could be dropped
    """
    data = {f"test{i}": [] for i in range(5)}

    for fold in data.keys():

        paths = glob(data_path + f"corpus/{fold}/*.split.txt")

        logging.info(f"\n{fold}: {len(paths)} documents")
        for txt_path in paths:
            ann_path = re.sub(".txt", ".ann", txt_path)

            doc_data, doc_entities, doc_relations = read_doc_annotation(txt_path, ann_path)

            for d in doc_data[discard_first_lines:]:
                # Discard empty sentences
                if len(d["tokens"]):
                    data[fold].append(d)

    for fold in data.keys():
        logging.info(f"\n{fold}")
        data_stats(data[fold])

    logging.info("TOTAL")
    data_stats(np.concatenate([v for v in data.values()]))

    return data, data_vocab(data)
