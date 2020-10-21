import logging
import os
import random
import shutil

import numpy as np
import torch


def set_random_seed(seed, cuda=True):
    np.random.seed(seed)  # cpu vars
    torch.manual_seed(seed)  # cpu  vars
    random.seed(seed)  # Python

    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, checkpoint='checkpoint.pth.tar', best='model_best.pth.tar'):
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, best)


def load_checkpoint(filename):
    if os.path.isfile(filename):
        logging.info("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=lambda storage, location: storage)
        epoch = checkpoint['epoch']
        logging.info("=> loaded checkpoint @ epoch {}".format(epoch))
    else:
        logging.info("=> no checkpoint found at '{}'".format(filename))
        return None

    return checkpoint


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(handler)

    # if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(log_path, "w", encoding="utf8")
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
