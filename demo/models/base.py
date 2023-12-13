import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, seq_len = 90, input_dim = 17*2, hidden_size=512, output_size=57):
        super(BaseModel, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_size = output_size

        self.lstm1 = nn.LSTM(input_dim, hidden_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.Linear(hidden_size * 2, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(256, output_size)

        self.act = torch.nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)

        out = self.dropout1(out)
        out = self.dense1(out)
        out = self.act(out)
        out = self.dropout2(out)
        out = self.dense2(out)

        return out
