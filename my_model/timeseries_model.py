from torch import nn


class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.linear1 = nn.Linear(hidden_size, 1)
        # self.linear2 = nn.Linear(sequence_length, 1)
        self.linear = nn.Linear(hidden_size * sequence_length, 1)


    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.flatten(1)
        out = self.linear(x)
        return out
