import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

batch_size = 1
hidden_size = 128
input_size = 32
output_size = 10
rnn = RNN(input_size, hidden_size, output_size)

sequence_length = 200

x = torch.randn(batch_size, input_size, requires_grad=True)
hidden = torch.randn(batch_size, hidden_size, requires_grad=True)
torch.onnx.export(rnn, (x, hidden), "rnn.onnx", export_params=True)

x = torch.randn(sequence_length, batch_size, input_size, requires_grad=True)
hidden = torch.randn(1, batch_size, hidden_size, requires_grad=True)

rnn_pytorch = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=False)
torch.onnx.export(rnn_pytorch, (x, hidden), "rnn_torch.onnx", export_params=False)
