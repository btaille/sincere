import sys

sys.path.append("../code")

from global_vars import DATA_DIR

import os
import logging
from data.dataset import dump
from data.ace_preprocess import load_ace05, load_ace04

logging.basicConfig(filename='ann2json.log', level=logging.INFO)

if os.path.exists("ace2005/corpus") and not os.path.exists(DATA_DIR + "ace05.json"):
    data, vocab = load_ace05("ace2005/")
    dump(data, DATA_DIR + "ace05.json")

if os.path.exists("ace2004/corpus") and not os.path.exists(DATA_DIR + "ace04.json"):
    data, vocab = load_ace04("ace2004/")
    dump(data, DATA_DIR + "ace04.json")
