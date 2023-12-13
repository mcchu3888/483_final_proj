import torch 
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv
import sys

class GAT_LSTM(nn.Module):
    def __init__(self, seq_len=90, num_input_keypoints=17, feature_dim=2, output_dim=57):
        super(GAT_LSTM, self).__init__()

        self.connections_2d = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
        self.edges_2d = torch.tensor(self.connections_2d, dtype=torch.long).t().contiguous()
        self.edges_2d = torch.cat([self.edges_2d, self.edges_2d.flip(0)], dim=1)
        self.seq_len = seq_len
        self.num_keypoints = num_input_keypoints
        self.feature_dim = feature_dim

        # GAT RESIDUAL BLOCKS
        self.gat1 = GATConv(self.feature_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.gat2 = GATConv(64, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.5)
        self.act = nn.ReLU()

        # LSTM LAYERS
        self.lstm1 = nn.LSTM(input_size=1122, hidden_size=512, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=1024, hidden_size=512, bidirectional=True, batch_first=True) 
        
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim))

    def forward(self, batch):
        x = self.gat1(batch.x, batch.edge_index)
        x = self.act(self.bn1(x))
        x = self.dropout(x)
        
        x = self.gat2(x, batch.edge_index)
        x = self.act(self.bn2(x))
        x = self.dropout(x)

        # skip connection
        x = torch.cat((x, batch.x), dim=1)

        # concatenate skeleton frames in the same batch for lstm
        x = x.view(-1, self.seq_len, self.num_keypoints, 64+self.feature_dim)
        embeddings = x.view(-1, self.seq_len, self.num_keypoints*(64+self.feature_dim))

        out, states = self.lstm1(embeddings)
        out, states = self.lstm2(out)
        out = self.head(out)
        return out
    
    def make_graph(self, x):
        x = x.view(-1, self.num_keypoints, self.feature_dim)
        graphs = [Data(x=graph, edge_index=self.edges_2d) for graph in x]
        batch = Batch().from_data_list(graphs)
        return batch