import json
import logging
from data.utils import data_stats
from data.dataset import data_vocab


def load_conll04(data_path):
    """Load CoNLL04 data formatted as in (Eberts 2019)'s code : https://github.com/markus-eberts/spert"""
    data = {}
    for split in ["train", "dev", "test"]:
        with open(data_path + f"conll04_{split}.json", "r") as file:
            data[split] = json.load(file)

    for split in data.keys():
        logging.info("\n" + split)
        data_stats(data[split])

    return data, data_vocab(data)
