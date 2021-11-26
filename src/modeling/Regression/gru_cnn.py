import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modeling.utils import PositionalEncoding, Flattening


class GRU_CNN(nn.Module):
    def __init__(self, len: int):
        super(GRU_CNN, self).__init__()

        self.seq_len = len
        self.embedding_dim = len + 1
        self.dropout_rate = 0.4
        self.RNN_hidden = 100

        self.embedding_layer = nn.Embedding(
            num_embeddings=4, embedding_dim=self.embedding_dim, max_norm=True
        )
        self.position_encoding = PositionalEncoding(
            dim=self.embedding_dim, max_len=self.seq_len, dropout=0.1
        )

        self.gru = nn.GRU(
            self.embedding_dim, self.RNN_hidden, num_layers=2, bidirectional=True
        )
        self.dropout = nn.Dropout()

        self.ConvLayer = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding="same", stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding="same", stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.flattening = Flattening()
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)

        self.predictor = nn.Sequential(
            nn.BatchNorm1d(5379),
            # nn.BatchNorm1d(2673),
            # nn.Linear(in_features=2673, out_features=512),
            nn.Linear(in_features=5379, out_features=512),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm1d(512),
            nn.Linear(in_features=512, out_features=32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, inputs):

        identity = inputs.clone()
        inputs_embs = self.embedding_layer(inputs) * math.sqrt(self.embedding_dim)
        # inputs_sums = self.position_encoding(inputs_embs)
        # output, _ = self.gru(inputs_sums)
        output, _ = self.gru(inputs_embs)

        F_RNN = output[:, :, : self.RNN_hidden]
        R_RNN = output[:, :, self.RNN_hidden :]
        output = torch.cat((F_RNN, R_RNN), 2)
        output = self.dropout(output)  # 256, 33, 200
        output = self.maxpool(output)
        output = self.flattening(output)  # 3267
        ##########################################################################

        conv_out = F.one_hot(identity).to(torch.float)
        conv_out = conv_out.transpose(1, 2)
        conv_out = self.ConvLayer(conv_out)
        conv_out = self.flattening(conv_out)
        # conv_out = self.maxpool(conv_out)

        # output = self.flattening(output)
        output = torch.cat((output, conv_out), dim=1)
        # output = output.reshape(output.shape[0], -1)
        output = self.predictor(output)

        return output.squeeze()
