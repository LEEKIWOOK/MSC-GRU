import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modeling.utils import PositionalEncoding, Flattening


class CNN(nn.Module):
    def __init__(self, len: int):
        super(CNN, self).__init__()

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
            nn.BatchNorm1d(2112),
            # nn.BatchNorm1d(2673),
            # nn.Linear(in_features=2673, out_features=512),
            nn.Linear(in_features=2112, out_features=512),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm1d(512),
            nn.Linear(in_features=512, out_features=32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, inputs):

        conv_out = F.one_hot(inputs).to(torch.float)
        conv_out = conv_out.transpose(1, 2)
        conv_out = self.ConvLayer(conv_out)
        conv_out = self.flattening(conv_out)
        # conv_out = self.maxpool(conv_out)

        # output = self.flattening(output)
        # output = output.reshape(output.shape[0], -1)
        output = self.predictor(conv_out)

        return output.squeeze()
