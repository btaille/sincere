import numpy as np
from tqdm import tqdm
import logging

def trim_embeddings(vocab, embedding_path, embedding_dim=300, init_std=0.001):
    """ Trim pretrained word embeddings to word vocabulary"""
    w2idx = vocab.words.val2idx
    size_vocab = len(w2idx)

    # Initialize random embeddings
    embeddings = np.random.normal(scale=init_std, size=(size_vocab, embedding_dim))

    # Populate with pretrained
    logging.info(f"Reading embedding file from {embedding_path}")
    found = 0
    with open(embedding_path, "r", encoding="utf8") as file:
        for i, line in enumerate(tqdm(file)):
            line = line.strip().split()
            if not len(line) == embedding_dim + 1:
                continue

            word, embedding = line[0], line[1:]
            if word in w2idx:
                found += 1
                embeddings[w2idx[word]] = embedding

    logging.info("Word embeddings trimmed. Found {} vectors for {} words".format(found, size_vocab))
    return embeddings
