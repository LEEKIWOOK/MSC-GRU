import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def dwise_conv(ch_in, stride=1):
    return nn.Sequential(
        # depthwise
        nn.Conv1d(
            ch_in,
            ch_in,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=ch_in,
            bias=False,
        ),
        nn.BatchNorm1d(ch_in),
        # nn.ReLU6(inplace=True),
        nn.ReLU(),
        nn.Dropout(0.3),
    )


def conv1x1(ch_in, ch_out):
    return nn.Sequential(
        nn.Conv1d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
        nn.BatchNorm1d(ch_out),
        # nn.ReLU6(inplace=True),
        nn.ReLU(),
        nn.Dropout(0.3),
    )


def conv3x3(ch_in, ch_out, stride):
    return nn.Sequential(
        nn.Conv1d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
        nn.BatchNorm1d(ch_out),
        # nn.ReLU6(inplace=True),
        nn.ReLU(),
        nn.Dropout(0.3),
    )


class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride == 1 and ch_in == ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend(
            [
                # dw
                dwise_conv(hidden_dim, stride=stride),
                # pw
                conv1x1(hidden_dim, ch_out),
            ]
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)


class MobileNetV2(nn.Module):
    def __init__(self, ch_in=4, n_classes=1):
        super(MobileNetV2, self).__init__()

        self.configs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            # [6, 64, 4, 2],
            # [6, 96, 3, 1],
            # [6, 160, 3, 2],
            # [6, 320, 1, 1],
        ]

        # self.stem_conv = conv3x3(ch_in, 32, stride=2)

        layers = []
        input_channel = 16
        self.embedding_layer = nn.Embedding(
            num_embeddings=4, embedding_dim=input_channel, max_norm=True
        )

        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(
                    InvertedBlock(
                        ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride
                    )
                )
                input_channel = c

        dropout_rate = 0.3

        self.layers = nn.Sequential(*layers)

        self.last_conv = conv1x1(input_channel, 288)

        # self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, n_classes))
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout_rate)
        self.flattening = nn.Flatten()
        self.predictor = nn.Sequential(
            # nn.BatchNorm1d(embedding_dim * 2 * seq_len),
            # nn.Linear(in_features=embedding_dim * 2 * seq_len, out_features=128),
            nn.BatchNorm1d(288),
            nn.Linear(in_features=288, out_features=32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, x):
        # x = F.one_hot(x).to(torch.float)
        # x = x.transpose(1, 2)
        # x = self.stem_conv(x)
        x = self.dropout(self.embedding_layer(x))  # 256, 33, 16
        x = x.transpose(1, 2)

        x = self.layers(x)
        x = self.last_conv(x)
        # x = self.flattening(x)
        x = self.predictor(x)
        # x = self.avg_pool(x).view(-1, 1280)
        # x = self.classifier(x)

        return x.squeeze()


if __name__ == "__main__":
    # model check
    model = MobileNetV2(ch_in=4, n_classes=1)
    # summary(model, (3, 224, 224), device='cpu')
