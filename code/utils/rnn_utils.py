from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class PackedRNN(nn.Module):
    """Wrapper for torch.nn.RNN to feed unordered and unpacked inputs"""

    def __init__(self, rnn):
        super(PackedRNN, self).__init__()
        self.rnn = rnn
        self.batch_first = self.rnn.batch_first

    def forward(self, inputs, seqlens, hidden=None):
        # Pack input sequence, apply RNN and Unpack output
        packed_inputs = pack_padded_sequence(inputs, seqlens, batch_first=self.batch_first,
                                             enforce_sorted=False)

        self.rnn.flatten_parameters()
        if hidden is None:
            packed_output, hidden = self.rnn(packed_inputs)
        else:
            packed_output, hidden = self.rnn(packed_inputs, hidden)

        output, _ = pad_packed_sequence(packed_output, batch_first=self.batch_first)

        return output, hidden