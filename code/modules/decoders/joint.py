from torch import nn


class JointDecoder(nn.Module):
    """Combination of several decoders for several tasks.
        Two modes in loss computation :
        1. (weigthed) sum of losses
        3. Individual loss given a task"""

    def __init__(self, decoders, loss_weights=None):
        super(JointDecoder, self).__init__()

        # init decoders
        self.decoders = {d.name: d for d in decoders}
        for name, decoder in self.decoders.items():
            self.add_module(name, decoder)

        # set weights
        if loss_weights is None:
            # default to uniform weights
            self.loss_weights = {d.name: 1 for d in decoders}
        else:
            self.loss_weights = {d.name: w for d, w in zip(decoders, loss_weights)}

    def forward(self, data, task="joint"):
        if task == "joint":
            # Pass through each decoder
            for name, decoder in self.decoders.items():
                decoder(data)

            # Compute each individual loss
            for name, decoder in self.decoders.items():
                if decoder.supervision in data.keys():
                    data.update({"{}_loss".format(name): decoder.loss(data)})

        else:
            assert task in self.decoders.keys()
            self.decoders[task](data)

    def loss(self, data, task="joint"):
        # Joint loss = weighted sum (1.)
        if task == "joint":

            # Compute each individual loss
            for name, decoder in self.decoders.items():
                if decoder.supervision in data.keys():
                    data.update({"{}_loss".format(name): decoder.loss(data)})

            # Weighted sum
            loss = 0
            for task, weight in self.loss_weights.items():
                loss += weight * data["{}_loss".format(task)]

        # Individual loss (2.)
        else:
            assert task in self.decoders.keys()
            loss = self.decoders[task].loss(data)
            data.update({"{}_loss".format(task): loss})

        data["loss"] = loss

        return loss
