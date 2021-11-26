import torch
import torch.nn as nn
from torchsummary import summary


class MobileNetV1(nn.Module):
    def __init__(self, ch_in=32):
        super(MobileNetV1, self).__init__()

        self.embedding_layer = nn.Embedding(
            num_embeddings=4, embedding_dim=ch_in, max_norm=True
        )
        self.dropout = nn.Dropout(0.3)

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv1d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm1d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv1d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm1d(inp),
                nn.ReLU(inplace=True),
                # pw
                nn.Conv1d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            # conv_dw(128, 256, 2),
            # conv_dw(256, 256, 1),
            # conv_dw(256, 512, 2),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 1024, 2),
            # conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding_layer(x)  # 256, 33, 16
        x = x.transpose(1, 2)

        x = self.model(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x


# if __name__ == "__main__":
#     # model check
#     model = MobileNetV1(ch_in=3, n_classes=1000)
#     summary(model, input_size=(3, 224, 224), device="cpu")
