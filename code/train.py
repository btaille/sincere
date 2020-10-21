import json
import logging
import os
from os.path import dirname

from torch import optim

from data.dataset import init_data_iterators, load_data, Dataset
from data.utils import data_stats
from global_vars import DATA_DIR
from models.ere import ERE
from utils.config import config_from_args
from utils.evaluation import re_score
from utils.train_utils import set_random_seed, set_logger, load_checkpoint

if __name__ == "__main__":
    # Load and check config from args
    config = config_from_args()

    if os.path.exists(config.run_dir + f"{config.criterion}_test_scores.json"):
        assert False, "Run already launched"

    # Set logger
    print("Logging in {}".format(os.path.join(config.run_dir, "train.log")))
    set_logger(os.path.join(config.run_dir, "train.log"))

    # Set random seed
    set_random_seed(config.seed)

    # Load data
    assert os.path.exists(DATA_DIR + f"{config.dataset}.json")
    data, vocab = load_data(DATA_DIR + f"{config.dataset}.json", verbose=False)

    # Standard mode = training on train set and validation on dev set
    train_key, dev_key, test_key = "train", "dev", "test"

    # Training on train + dev => no validation
    if config.train_mode == "train+dev":
        train_key, dev_key, test_key = "train+dev", None, "test"

        # If a standard training was done with same config get best number of epochs on dev
        best_checkpoint_path = os.path.join(dirname(dirname(config.run_dir)), f"{config.criterion}_best.pth.tar")
        if os.path.exists(best_checkpoint_path):
            best_checkpoint = load_checkpoint(best_checkpoint_path)
            config.epochs = best_checkpoint["epoch"]
            logging.info(f"Setting n epochs to {config.epochs}")

        data["train+dev"] = data["train"] + data["dev"]
        del data["train"], data["dev"]

    # Log Dataset statistics
    for k in data.keys():
        logging.info("\n" + k)
        data_stats(data[k])

    # Init data loaders
    datasets = {split: Dataset(data[split], vocab) for split in data.keys()}
    for d in datasets.values():
        d.prepare_dataset(config)

    iterators = {task: init_data_iterators(datasets, batch_size=config.batch_size,
                                           train_key=train_key, shuffle_train=True) for task in config.tasks}

    #### Model Initialization
    model = ERE.from_config(config, vocab)
    model.to(config.device)

    logging.info(model)

    ### Init Optimizers
    optimizers = {task: optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate,
                                   betas=(0.9, 0.999),
                                   eps=1e-8, weight_decay=0) for task in config.tasks}

    ### Train model
    logging.info("Training model in {}".format(config.run_dir))

    # Reload checkpoint if exist
    tasks = config.tasks
    epoch_start = 0
    best_f1 = {t: 0 for t in tasks}
    epochs_no_improv = {t: 0 for t in tasks}
    best_score = None

    if os.path.exists(config.run_dir + f"{config.criterion}_checkpoint.pth.tar"):
        logging.info("Reload model...")
        checkpoint = load_checkpoint(config.run_dir + f"{config.criterion}_checkpoint.pth.tar")

        epoch_start = checkpoint["epoch"]
        epochs_no_improv = checkpoint["epochs_no_improv"]

        if "scores" in checkpoint:  # no scores recorded when training without validation
            best_score = checkpoint["scores"]
            best_f1 = {t: checkpoint["scores"][t]["ALL"]["f1"] for t in best_score}

        for task in optimizers.keys():
            optimizers[task].load_state_dict(checkpoint["optimizers"][task])

        model.load_state_dict(checkpoint["model"])

    ### Training task or Joint Training method
    if len(tasks) == 1:
        task = tasks[0]
    else:
        task = "joint"

    ### Train Loop
    model.train_loop(iterators, optimizers, config.run_dir, task=task,
                     epochs=config.epochs, epoch_start=epoch_start, min_epochs=config.min_epochs,
                     patience=config.patience, epochs_no_improv=epochs_no_improv,
                     criterion=config.criterion, best_f1=best_f1,
                     train_key=train_key, dev_key=dev_key,
                     tensorboard_summary=config.tensorboard)

    ### Reload best DEV model and Evaluate on DEV and TEST
    best_checkpoint = load_checkpoint(config.run_dir + f"{config.criterion}_best.pth.tar")
    model.load_state_dict(best_checkpoint["model"])

    if "ner" in config.tasks:
        if dev_key is not None:
            ner_dev_preds, _, ner_dev_loss, ner_dev_scores = model.evaluate_ner(iterators["ner"][dev_key])

            with open(config.run_dir + f"ner_{dev_key}_preds.json", "w") as file:
                json.dump(ner_dev_preds, file)
            with open(config.run_dir + f"ner_{dev_key}_scores.json", "w") as file:
                json.dump(ner_dev_scores, file)

        ner_test_preds, _, ner_test_loss, ner_test_scores = model.evaluate_ner(iterators["ner"][test_key])

        with open(config.run_dir + f"ner_{test_key}_preds.json", "w") as file:
            json.dump(ner_test_preds, file)
        with open(config.run_dir + f"ner_{test_key}_scores.json", "w") as file:
            json.dump(ner_test_scores, file)

    if "re" in config.tasks:
        if dev_key is not None:
            re_dev_preds, re_dev_gt, re_dev_loss, re_dev_scores = model.evaluate_re(iterators["re"][dev_key],
                                                                                    mode="strict")

            with open(config.run_dir + f"re_{dev_key}_preds.json", "w") as file:
                json.dump(re_dev_preds, file)
            with open(config.run_dir + f"re_{dev_key}_scores.json", "w") as file:
                json.dump(re_dev_scores, file)
            with open(config.run_dir + f"re_{dev_key}_scores_boundaries.json", "w") as file:
                json.dump(re_score(re_dev_preds, re_dev_gt, model.decoder.decoders["re"].vocab, mode="boundaries"),
                          file)

        re_test_preds, re_test_gt, re_test_loss, re_test_scores = model.evaluate_re(iterators["re"][test_key],
                                                                                    mode="strict")

        with open(config.run_dir + f"re_{test_key}_preds.json", "w") as file:
            json.dump(re_test_preds, file)
        with open(config.run_dir + f"re_{test_key}_scores.json", "w") as file:
            json.dump(re_test_scores, file)
        with open(config.run_dir + f"re_{test_key}_scores_boundaries.json", "w") as file:
            json.dump(re_score(re_test_preds, re_test_gt, model.decoder.decoders["re"].vocab, mode="boundaries"),
                      file)
