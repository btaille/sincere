import os
import torch

import json
import argparse

from global_vars import RUN_DIR
from global_vars import EMBEDDINGS_DIR


def config_from_args(arg_list=None):
    """ Argument Parser """
    parser = argparse.ArgumentParser()

    # Parameter in json to reload an already define config
    parser.add_argument("-pf", "--parameter_file", help="parameter json file", default=None)

    # From command_line
    # Training hyperparameters
    parser.add_argument("-ds", "--dataset", help="dataset", default="conll04")
    parser.add_argument("-t", "--tasks", help="tasks", nargs="+", default=["ner", "re"])
    parser.add_argument("-m", "--train_mode", help="'train' or 'train+dev'", default="train")

    parser.add_argument("-s", "--seed", type=int, help="torch manual random seed", default=0)
    parser.add_argument("-ep", "--epochs", type=int, help="max number of epochs", default=100)
    parser.add_argument("-p", "--patience", type=int, help="patience", default=5)
    parser.add_argument("-min", "--min_epochs", type=int, help="min number of epochs", default=10)

    parser.add_argument("-tb", "--tensorboard", type=int, help="whether to log a tensorboard summary", default=1)

    parser.add_argument("-lr", "--learning_rate", type=float, help="learning rate", default=1e-3)
    parser.add_argument("-bs", "--batch_size", type=int, help="batch size", default=1)
    parser.add_argument("-d", "--dropout", type=float, help="dropout", default=0.1)
    parser.add_argument("-dev", "--device", help="pytorch device", default="cuda")

    # Model Architecture
    parser.add_argument("-emb", "--embedder", help="embedder list", nargs="+", default=["bert-base"])
    parser.add_argument("-enc", "--encoder", help="encoder", default=None)
    parser.add_argument("-ner_dec", "--ner_decoder", help="ner decoder", default="iobes")

    #### Embedders
    # Char Embedder
    parser.add_argument("-ce", "--char_dim", type=int, help="dimension of char embeddings", default=100)
    parser.add_argument("-ch", "--char_hidden", type=int, help="dimension of char hidden layer", default=25)
    parser.add_argument("-cp", "--char_pool", help="pooling for the char-level encoder", default="last")

    # Word Embedder
    parser.add_argument("-wp", "--word_path", help="path of pretrained word embeddings",
                        default=EMBEDDINGS_DIR + "glove.840B/glove.840B.300d.txt")
    parser.add_argument("-we", "--word_dim", type=int, help="dimension of word embeddings", default=300)
    parser.add_argument("-wd", "--word_dropout", type=float, help="word_dropout", default=0.1)
    parser.add_argument("-f", "--freeze", type=int, help="freeze embeddings", default=0)

    # BERT Embedder
    parser.add_argument("-bft", "--bert_finetune", help="finetune BERT", type=int, default=1)
    parser.add_argument("-bpool", "--bert_pool", help="pooling of subwords", default="max")

    #### BiLSTM Encoder
    parser.add_argument("-lh", "--bilstm_hidden", type=int, help="dimension of bilstm hidden layer", default=384)
    parser.add_argument("-ll", "--bilstm_layers", type=int, help="num bilstm layers", default=1)

    ### NER Decoder
    parser.add_argument("-ner_max_span", "--ner_max_span", type=int, help="Max considered span length in Span NER",
                        default=10)
    parser.add_argument("-ner_ns", "--ner_negative_sampling", type=int, help="Negative sampling in Span NER",
                        default=100)
    parser.add_argument("-ner_span_emb", "--ner_span_embedding_dim", type=int,
                        help="Span length embedding dim in Span NER", default=25)
    parser.add_argument("-pool", "--pool_fn", help="Pooling function for Span Representations and Context",
                        default="max")

    #### RE Decoder
    parser.add_argument("-re_ns", "--re_negative_sampling", type=int, help="Negative sampling in Span RE",
                        default=100)
    parser.add_argument("-re_biaffine", "--re_biaffine", type=int, help="Add bilinear term in RE", default=0)
    parser.add_argument("-re_context", "--re_context", type=int, help="Use middle context max pooling in RE", default=1)

    ### Joint decoding
    parser.add_argument("-crt", "--criterion", help="early stopping criterion", default="re")

    ### Run dir Prefix and Suffix
    parser.add_argument("-pfx", "--prefix", help="add a prefix to run dir", default="")
    parser.add_argument("-sfx", "--suffix", help="add a suffix to run dir", default="")

    """ Convert arg_list into dictionary """
    if arg_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arg_list)

    if args.parameter_file is not None:
        return Config.from_saved(args.parameter_file)
    else:
        return Config(args)


class Config:
    def __init__(self, args):
        if args is not None:
            self.dataset = args.dataset

            self.embedder = args.embedder
            self.encoder = args.encoder
            self.ner_decoder = args.ner_decoder

            self.seed = args.seed
            self.epochs = args.epochs
            self.patience = args.patience
            self.min_epochs = args.min_epochs

            self.train_mode = args.train_mode
            self.tasks = args.tasks
            self.tensorboard = args.tensorboard

            self.learning_rate = args.learning_rate
            self.batch_size = args.batch_size
            self.dropout = args.dropout

            self.device = args.device

            # Embedders
            if "char" in self.embedder:
                self.char_embedding_dim = args.char_dim
                self.char_hidden_dim = args.char_hidden
                self.char_pool = args.char_pool

            if "word" in self.embedder:
                self.word_embedding_path = args.word_path
                self.word_embedding_dim = args.word_dim
                self.word_dropout = args.word_dropout
                self.word_freeze = args.freeze

            if "bert-base" in self.embedder or "bert-large" in self.embedder:
                self.bert_finetune = args.bert_finetune
                self.bert_pool = args.bert_pool

            # Encoders
            if self.encoder == "bilstm":
                self.bilstm_layers = args.bilstm_layers
                self.bilstm_hidden = args.bilstm_hidden

            self.pool_fn = args.pool_fn

            # NER Decoder
            if self.ner_decoder == "span":
                self.ner_negative_sampling = args.ner_negative_sampling
                self.ner_max_span = args.ner_max_span
                self.ner_span_emb = args.ner_span_embedding_dim

            # RE Decoder
            if "re" in self.tasks:
                self.re_biaffine = args.re_biaffine
                self.re_context = args.re_context
                self.re_negative_sampling = args.re_negative_sampling

            # Joint Decoder
            self.criterion = args.criterion
            assert self.criterion in self.tasks

            # RUN DIR
            self.run_dir = format_run_dir(self, args.prefix, args.suffix)

            # Check validity of config
            check_config(self)

            # Dump config to json
            if not os.path.exists(self.run_dir):
                os.makedirs(self.run_dir)
            self.to_json(os.path.join(self.run_dir, "config.json"))

    def to_json(self, json_path):
        with open(json_path, "w") as file:
            json.dump(self.__dict__, file)

    @classmethod
    def from_json(cls, json_path):
        config = cls(None)
        with open(json_path, "r") as file:
            config.__dict__ = json.load_file


def format_run_dir(config, prefix="", suffix=""):
    # Format run directory
    if not "run_dir" in config.__dict__:
        run_dir = f"{config.dataset}_{'-'.join(sorted(config.embedder))}"

        if config.encoder == "bilstm":
            run_dir += f"_bilstm-{config.bilstm_layers}-{config.bilstm_hidden}"

        if "ner" in config.tasks:
            run_dir += f"_ner-{config.ner_decoder}"

        if "re" in config.tasks:
            run_dir += "_re"
            if config.re_biaffine:
                run_dir += "-biaff"
            if config.re_context:
                run_dir += "-ctxt"

        run_dir += f"_lr-{config.learning_rate}_bs-{config.batch_size}_d-{config.dropout}"

        if config.device == "cpu":
            run_dir += "_cpu"

        if config.criterion != "re" and len(config.tasks) >= 2:
            run_dir += f"_crt-{config.criterion}"

        if len(prefix):
            run_dir = f"{prefix}_" + run_dir

        if len(suffix):
            run_dir += f"_{suffix}"

        run_dir += f"/seed_{config.seed}/"

        if config.train_mode == "train+dev":
            run_dir += "train+dev/"

        return os.path.join(RUN_DIR, run_dir)


def check_config(config):
    """Assert parameters are a valid set and format training folder."""

    # Check config
    assert config.dataset in ["conll04", "ace05"]
    assert config.train_mode in ["train", "train+dev"]

    for emb in config.embedder:
        assert emb in ["word", "char", "bert-base", "bert-large"], emb

    if "char" in config.embedder:
        assert config.char_pool in ["last", "avg", "max"]

    if config.encoder is not None:
        assert config.encoder == "bilstm"

    for task in config.tasks:
        assert task in ["ner", "re"]

    assert config.ner_decoder in ["iobes", "span"]

    if "cuda" in config.device:
        assert torch.cuda.is_available(), "CUDA not available"

