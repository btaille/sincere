import logging

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils.evaluation import ner_score, re_score
from utils.train_utils import save_checkpoint

try:
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD = True
except ImportError:
    TENSORBOARD = False


def add_score(writer, score, n_iter, task="ner"):
    """Add score results to Tensorboard writer"""
    writer.add_scalars("{}/f1".format(task), {ent: score[ent]["f1"] for ent in score.keys()}, n_iter)
    writer.add_scalars("{}/p".format(task), {ent: score[ent]["p"] for ent in score.keys()}, n_iter)
    writer.add_scalars("{}/r".format(task), {ent: score[ent]["r"] for ent in score.keys()}, n_iter)


class Model(nn.Module):
    def train_step(self, batch, optimizer, task="ner", optimizer_step=True, gradient_accumulation=1,
                   **kwargs):
        self.forward(batch, task, **kwargs)
        loss = self.loss(batch, task) / gradient_accumulation

        if loss.item():  # do not backward if loss == 0 (e.g. when there are no pairs of entities in RE)
            loss.backward()
            if optimizer_step:
                optimizer.step()
                self.zero_grad()

    def run_epoch(self, iterators, epoch, optimizers, writer, task="joint", train_key="train",
                  gradient_accumulation=1, **kwargs):
        self.train()
        for iterator in iterators.values():
            iterator[train_key].reinit()

        tasks = iterators.keys()
        losses = {t: [] for t in tasks}

        #### Select tasks for each batch in the epoch
        # If task is "joint" : for each batch all tasks will be performed and the loss is a (weighted) sum
        if task == "joint":
            step_task = list(tasks)[0]
            n_batches = len(iterators[step_task][train_key])

        # If task is one task : the model is only trained for this task
        else:
            assert task in tasks
            step_task = task
            n_batches = len(iterators[step_task][train_key])

        #### Perform the training_steps
        for i in tqdm(range(n_batches)):
            n_iter = epoch * n_batches + i

            batch = next(iterators[step_task][train_key])

            self.train_step(batch, optimizers[step_task], task=step_task if not task == "joint" else "joint",
                            optimizer_step=(i + 1) % gradient_accumulation == 0,
                            gradient_accumulation=gradient_accumulation, **kwargs)

            for t in tasks:
                if "{}_loss".format(t) in batch.keys():
                    losses[t].append(batch["{}_loss".format(t)].item())
                    if writer is not None:
                        writer.add_scalars("{}_loss".format(t), {"train": batch["{}_loss".format(t)].item()}, n_iter)

            if "loss" in batch.keys() and writer is not None:
                writer.add_scalars("loss", {"train": batch["loss"].item()}, n_iter)

        return losses

    def train_loop(self, iterators, optimizers, run_dir, task="joint",
                   epochs=100, min_epochs=0, patience=5, epoch_start=0,
                   best_f1=None, epochs_no_improv=None, best_scores=None,
                   criterion="re", mode="strict", train_key="train", dev_key="dev",
                   save_all_tasks=False, gradient_accumulation=1, tensorboard_summary=True, **kwargs):

        # Validation or not
        if dev_key is not None:
            logging.info("Starting train loop: {} epochs; {} min; {} patience".format(epochs, min_epochs, patience))
        else:
            logging.info("Starting train loop without validation for {} epochs".format(epochs))
            patience = 0
            min_epochs = epochs

        #
        tasks = iterators.keys()
        if best_f1 is None:
            best_f1 = {t: 0 for t in tasks}

        if epochs_no_improv is None:
            epochs_no_improv = {t: 0 for t in tasks}

        if best_scores is None:
            best_scores = {t: 0 for t in tasks}

        # Check for early stopping already matched (when reloading a checkpoint)
        if patience and epoch_start > min_epochs and epochs_no_improv[criterion] >= patience:
            logging.info("Early stopping after {} epochs without improvement.".format(patience))
        else:
            writer = SummaryWriter(run_dir) if TENSORBOARD and tensorboard_summary else None
            # Training loop
            for epoch in range(epoch_start, epochs):
                logging.info("Epoch {}/{} :".format(epoch + 1, epochs))
                train_losses = self.run_epoch(iterators, epoch, optimizers, writer, task=task,
                                              train_key=train_key,
                                              gradient_accumulation=gradient_accumulation)
                n_iter = (epoch + 1) * len(list(train_losses.values())[0])

                # Log train losses + evaluate on dev if not None
                if "ner" in tasks:
                    logging.info("Train NER Loss : {}".format(np.mean(train_losses["ner"])))
                    if dev_key is not None:
                        ner_preds, _, ner_loss, ner_scores = self.evaluate_ner(iterators["ner"][dev_key])
                        logging.info("Dev NER Loss : {}".format(ner_loss))

                if "re" in tasks:
                    logging.info("Train RE Loss : {}".format(np.mean(train_losses["re"])))
                    if dev_key is not None:
                        re_preds, _, re_loss, re_scores = self.evaluate_re(iterators["re"][dev_key], mode=mode)
                        logging.info("Dev RE Loss : {}".format(re_loss))

                # If validation : record current and best checkpoints + enable early stopping on dev score
                if dev_key is not None:
                    # save checkpoint and scores
                    scores = {}
                    f1 = {}
                    for t in tasks:
                        f1[t] = locals()["{}_scores".format(t)]["ALL"]["f1"]
                        scores[t] = locals()["{}_scores".format(t)]

                    for t in f1.keys():
                        if f1[t] > best_f1[t] or epoch == 0:
                            logging.info("New best {} F1 score on dev : {}".format(t, f1[t]))
                            if save_all_tasks or t == criterion:
                                logging.info("Saving model...")
                            best_f1[t] = f1[t]
                            epochs_no_improv[t] = 0
                            is_best = True

                        else:
                            epochs_no_improv[t] += 1
                            is_best = False

                        state = {'epoch': epoch + 1,
                                 'epochs_no_improv': epochs_no_improv,
                                 'model': self.state_dict(),
                                 'scores': scores,
                                 'optimizers': {k: optimizer.state_dict() for k, optimizer in optimizers.items()}
                                 }

                        if save_all_tasks or t == criterion:
                            save_checkpoint(state, is_best, checkpoint=run_dir + '{}_checkpoint.pth.tar'.format(t),
                                            best=run_dir + '{}_best.pth.tar'.format(t))

                    if TENSORBOARD and tensorboard_summary:
                        if "ner" in iterators.keys():
                            writer.add_scalars("ner_loss", {"dev": ner_loss}, n_iter)
                            add_score(writer, ner_scores, n_iter, task="ner")
                        if "re" in iterators.keys():
                            writer.add_scalars("re_loss", {"dev": re_loss}, n_iter)
                            add_score(writer, re_scores, n_iter, task="re")

                    # early stopping
                    if patience and epoch > min_epochs and epochs_no_improv[criterion] >= patience:
                        logging.info(
                            "Early stopping after {} epochs without improvement on {}.".format(patience, criterion))
                        break

                # Else : record current checkpoint
                else:
                    state = {'epoch': epoch + 1,
                             'epochs_no_improv': 0,
                             'model': self.state_dict(),
                             'optimizers': {k: optimizer.state_dict() for k, optimizer in optimizers.items()}
                             }

                    save_checkpoint(state, is_best=epoch == epochs - 1,
                                    checkpoint=run_dir + '{}_checkpoint.pth.tar'.format(criterion),
                                    best=run_dir + '{}_best.pth.tar'.format(criterion))

            if TENSORBOARD and tensorboard_summary:
                writer.close()

    def evaluate_ner(self, iterator):
        """Pass through the model in inference mode and reformat outputs to use ner_score"""
        losses = []
        pred_entities = []
        gt_entities = []
        self.eval()
        iterator.reinit()
        with torch.no_grad():
            for batch in tqdm(iterator):
                self.forward(batch, task="ner")
                gt_entities.extend(batch["entities"])
                pred_entities.extend(
                    [[{"start": span[0], "end": span[1], "type": ent_type} for span, ent_type in s.items()] for s in
                     batch["pred_entities"]])
                losses.append(self.decoder.loss(batch, "ner").item())

        return pred_entities, gt_entities, np.mean(losses), ner_score(pred_entities, gt_entities,
                                                                      self.decoder.decoders["ner"].vocab)

    def evaluate_re(self, iterator, mode="strict"):
        """Pass through the model in inference mode and reformat outputs to use re_score"""
        losses = []
        pred_relations = []
        gt_relations = []
        self.eval()
        iterator.reinit()
        with torch.no_grad():
            for batch in tqdm(iterator):
                self.forward(batch, task="re")

                pred_relations.extend(batch["pred_relations"])

                losses.append(self.decoder.loss(batch, "re").item())

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

        return pred_relations, gt_relations, np.mean(losses), re_score(pred_relations, gt_relations,
                                                                       self.decoder.decoders["re"].vocab,
                                                                       mode=mode)


class EmbedderEncoderDecoder(Model):
    def __init__(self, embedder, encoder, decoder):
        super(EmbedderEncoderDecoder, self).__init__()
        # parameters
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data, task="re"):
        if self.encoder is not None:
            data.update({"embeddings": self.embedder(data)["embeddings"]})
            data.update({"encoded": self.encoder(data)["output"]})
        else:
            data.update({"encoded": self.embedder(data)["embeddings"]})

        self.decoder(data, task)
        return data

    def loss(self, data, task="ner"):
        return self.decoder.loss(data, task)
