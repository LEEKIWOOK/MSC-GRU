import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, len: int):
        super(CNN, self).__init__()

        seq_len = len
        embedding_dim = 64
        dropout_rate = 0.3

        self.embedding_layer = nn.Embedding(
            num_embeddings=4, embedding_dim=embedding_dim, max_norm=True
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.ConvLayer1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=2),
            nn.Conv1d(
                embedding_dim,
                embedding_dim * 2,
                kernel_size=1,
                padding="same",
                stride=1,
                # bias=T,
            ),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.MaxPool1d(kernel_size=3, stride=2),
            # nn.AvgPool1d(kernel_size=3, stride=2),
        )

        self.ConvLayer2 = nn.Sequential(
            nn.Conv1d(
                embedding_dim,
                embedding_dim * 2,
                kernel_size=1,
                padding="same",
                stride=1,
                # bias=False,
            ),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.MaxPool1d(kernel_size=3, stride=2),
            # nn.AvgPool1d(kernel_size=3, stride=2),
        )
        self.ConvLayer3 = nn.Sequential(
            nn.Conv1d(
                embedding_dim,
                embedding_dim * 2,
                kernel_size=1,
                padding="same",
                stride=1,
                # bias=False,
            ),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(
                embedding_dim * 2,
                embedding_dim * 2,
                kernel_size=3,
                padding="same",
                stride=1,
                # bias=False,
            ),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.MaxPool1d(kernel_size=3, stride=2),
            # nn.AvgPool1d(kernel_size=3, stride=2),
        )
        self.ConvLayer4 = nn.Sequential(
            nn.Conv1d(
                embedding_dim,
                int(embedding_dim * 1.5),
                kernel_size=1,
                padding="same",
                stride=1,
                # bias=False,
            ),
            nn.BatchNorm1d(int(embedding_dim * 1.5)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(
                int(embedding_dim * 1.5),
                embedding_dim * 2,
                kernel_size=3,
                padding="same",
                stride=1,
                # bias=False,
            ),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(
                embedding_dim * 2,
                embedding_dim * 2,
                kernel_size=3,
                padding="same",
                stride=1,
                # bias=False,
            ),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.MaxPool1d(kernel_size=3, stride=2),
            # nn.AvgPool1d(kernel_size=3, stride=2),
        )
        self.flattening = nn.Flatten()

        self.predictor = nn.Sequential(
            # nn.BatchNorm1d(embedding_dim * 2 * seq_len),
            # nn.Linear(in_features=embedding_dim * 2 * seq_len, out_features=128),
            nn.BatchNorm1d(7040),
            nn.Linear(in_features=7040, out_features=512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(512),
            nn.Linear(in_features=512, out_features=64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, inputs):

        embs1 = self.dropout(self.embedding_layer(inputs))
        embs1 = embs1.transpose(1, 2)
        embs2 = embs1.clone()
        embs3 = embs1.clone()
        embs4 = embs1.clone()

        conv1 = self.ConvLayer1(embs1)
        conv2 = self.ConvLayer2(embs2)
        conv3 = self.ConvLayer3(embs3)
        conv4 = self.ConvLayer4(embs4)

        # Case 1 : mean : 2, 3, 4 = 0.8199963803803648
        # conv_out = torch.mean(torch.stack([conv2, conv3, conv4]), dim=0)

        # Case 2 : cat : 0.8218381793350873 -> embedding dim : 16
        conv_out = self.dropout(torch.cat([conv1, conv2, conv3, conv4], dim=2))

        # Case 3 : cat : 0.8226870731397585 -> embedding dim : 64

        # Case 4 : maxpooling 추가 -> 0.83745615092924

        conv_out = self.flattening(conv_out)
        conv_out = self.predictor(conv_out)

        return conv_out.squeeze()
