from torch import nn


class StreamableLSTM(nn.Module):
    """LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """

    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True, bidirectional: bool = False):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        self.projection = nn.Linear(2 * dimension, dimension) if bidirectional else None

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.bidirectional:
            y = self.projection(y)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y
