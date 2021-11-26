import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modeling.utils import PositionalEncoding, Flattening


class MHSA(nn.Module):
    def __init__(self, len: int):
        super(MHSA, self).__init__()

        self.seq_len = len
        self.embedding_dim = 128
        self.dropout_rate = 0.4
        self.single_head_size = 128
        self.multi_head_num = 8
        self.multi_head_size = 100  ###

        self.embedding_layer = nn.Embedding(
            num_embeddings=4, embedding_dim=self.embedding_dim, max_norm=True
        )
        self.position_encoding = PositionalEncoding(
            dim=self.embedding_dim, max_len=self.seq_len, dropout=0.1
        )

        self.Q = nn.ModuleList(
            [
                nn.Linear(
                    in_features=self.embedding_dim, out_features=self.single_head_size
                )
                for i in range(0, self.multi_head_num)
            ]
        )
        self.K = nn.ModuleList(
            [
                nn.Linear(
                    in_features=self.embedding_dim, out_features=self.single_head_size
                )
                for i in range(0, self.multi_head_num)
            ]
        )
        self.V = nn.ModuleList(
            [
                nn.Linear(
                    in_features=self.embedding_dim, out_features=self.single_head_size
                )
                for i in range(0, self.multi_head_num)
            ]
        )

        self.relu = nn.ModuleList([nn.ReLU() for i in range(0, self.multi_head_num)])
        self.MultiHeadLinear = nn.Sequential(
            nn.LayerNorm(self.single_head_size * self.multi_head_num),
            nn.Linear(
                in_features=self.single_head_size * self.multi_head_num,
                out_features=self.multi_head_size,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.flattening = Flattening()
        self.predictor = nn.Sequential(
            nn.BatchNorm1d(1617),
            nn.Linear(in_features=1617, out_features=512),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm1d(512),
            nn.Linear(in_features=512, out_features=32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(in_features=32, out_features=1),
        )

    def attention(self, query, key, value, mask=None, dropout=0.0):
        # based on: https://nlp.seas.harvard.edu/2018/04/03/attention.html
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, inputs):

        identity = inputs.clone()

        inputs_embs = self.embedding_layer(inputs) * math.sqrt(self.embedding_dim)
        inputs_sums = self.position_encoding(inputs_embs)
        output = inputs_sums
        ##########################################################################

        pAttn_concat = torch.Tensor([]).to(inputs.device)
        attn_concat = torch.Tensor([]).to(inputs.device)
        for i in range(0, self.multi_head_num):
            query = self.Q[i](output)
            key = self.K[i](output)
            value = self.V[i](output)
            attnOut, p_attn = self.attention(query, key, value, dropout=0.0)
            attnOut = self.relu[i](attnOut)
            attn_concat = torch.cat((attn_concat, attnOut), dim=2)

        attn_out = self.MultiHeadLinear(attn_concat)
        attn_out = self.maxpool(attn_out)
        attn_out = self.flattening(attn_out)

        output = self.predictor(attn_out)
        return output.squeeze()