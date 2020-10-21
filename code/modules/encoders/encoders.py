import torch
from torch import nn
from utils.rnn_utils import PackedRNN


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirectional=True, dropout=0., n_layers=1):
        super(BiLSTMEncoder, self).__init__()
        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)

        self.hidden_dim = hidden_dim
        self.output_dim = 2 * self.hidden_dim if bidirectional else self.hidden_dim

        self.n_layers = n_layers

        self.layers = [PackedRNN(nn.LSTM(input_dim, hidden_dim, bidirectional=bidirectional, batch_first=True))]

        for _ in range(self.n_layers - 1):
            self.layers.append(
                PackedRNN(nn.LSTM(self.output_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)))

        self.name = "bilstm"

        for i, layer in enumerate(self.layers):
            self.add_module("lstm_{}".format(i), layer)

    def forward(self, data, input_key="embeddings", seqlen_key="n_words"):
        inputs = data[input_key]
        seqlens = data[seqlen_key]

        for l in self.layers:
            inputs = self.drop(inputs)
            inputs, self.hidden = l(inputs, seqlens)

        # Final hidden state
        final_hidden = torch.cat([self.hidden[0][0], self.hidden[0][1]], dim=-1)
        final_cell = torch.cat([self.hidden[1][0], self.hidden[1][1]], dim=-1)

        return {"output": inputs, "last_hidden": final_hidden, "last_cell": final_cell}
