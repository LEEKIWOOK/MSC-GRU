import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.autograd import Variable

# from modeling.utils import PositionalEncoding, Flattening
import matplotlib.pyplot as plt

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels=4, emb_size=64, patch_size=4):
#         super().__init__()
#         self.patch_size = patch_size
#         self.seqlen = 33

#         self.proj = nn.Conv1d(4, 64, kernel_size=3, stride=3, padding=0)
#         self.rearrange = Rearrange("b e (w) -> b (w) e")
#         # self.norm = norm_layer(emb_size / 3 + 1)
#         # self.projection = nn.Sequential(
#         #     nn.Conv1d(in_channels, emb_size, kernel_size=3, stride=3, padding=0),
#         #     Rearrange("b e (w) -> b (w) e"),
#         # )
#         self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
#         self.positions = nn.Parameter(
#             torch.randn((self.seqlen // patch_size) ** 2 + 1, emb_size)
#         )

#     def forward(self, x):
#         b, _, _ = x.shape
#         # x = self.projection(x)
#         x = self.proj(x)
#         x = self.rearrange(x)

#         cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=b)

#         x = torch.cat([cls_tokens, x], dim=1)
#         x += self.positions
#         return x


class PatchEmbedding(nn.Module):
    def __init__(self, embedding_size, batch_size):
        super().__init__()

        self.patch_size = 8
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        self.projection = nn.Linear(33, self.embedding_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_size))
        self.positions = nn.Parameter(torch.randn(4 + 1, self.embedding_size))

    def forward(self, x):
        input_encd = F.one_hot(x).to(torch.float)
        input_encd = input_encd.transpose(1, 2)

        x = self.projection(input_encd)
        cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=self.batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        patch_embd = x + self.positions
        return patch_embd


class MHA(nn.Module):
    def __init__(self, embedding_size: int, multi_head_num: int, drop_p=0.0):
        super().__init__()

        self.embedding_size = embedding_size
        self.multi_head_num = multi_head_num
        self.drop_p = drop_p

        self.keys = nn.Linear(self.embedding_size, self.embedding_size)
        self.queries = nn.Linear(self.embedding_size, self.embedding_size)
        self.values = nn.Linear(self.embedding_size, self.embedding_size)

        self.att_drop = nn.Dropout(self.drop_p)
        self.projection = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, x, mask=None):

        q = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.multi_head_num)
        k = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.multi_head_num)
        v = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.multi_head_num)

        energy = torch.einsum("bhqd, bhkd -> bhqk", q, k)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.embedding_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        out = torch.einsum("bhal, bhlv -> bhav", att, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, embedding_size, expansion: int, drop_p: float):
        super().__init__(
            nn.Linear(embedding_size, expansion * embedding_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * embedding_size, embedding_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self, embedding_size: int, drop_p=0.0, expansion=4, multi_head_num=8, **kwargs
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embedding_size),
                    MHA(embedding_size, multi_head_num=multi_head_num),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embedding_size),
                    FeedForwardBlock(
                        embedding_size, expansion=expansion, drop_p=drop_p
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, embedding_size: int, **kwargs):
        super().__init__(
            *[TransformerEncoderBlock(embedding_size, **kwargs) for _ in range(depth)]
        )


class Predictor(nn.Sequential):
    def __init__(self, embedding_size: int):
        super().__init__(
            Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, 1),
        )


class ViT(nn.Sequential):
    def __init__(self, embedding_size: int, depth=4, batch_size=256, **kwargs):
        super(ViT, self).__init__(
            PatchEmbedding(embedding_size, batch_size),
            TransformerEncoder(depth, embedding_size, **kwargs),
            Predictor(embedding_size),
        )

        # self.seq_len = len
        # self.batch_size = batch_size

        # self.multi_head_num = 8

        # self.dropout_rate = 0.4
        # self.single_head_size = 32

        # self.multi_head_size = 100  ###

        # self.patchembedding = nn.Sequential(
        #     nn.Conv1d(4, self.embedding_dim, kernel_size=3, stride=3, padding=0),
        # Rearrange("b e (w) -> b (w) e"),
        # self.norm = nn.LayerNorm(11)

        # self.embedding_layer = nn.Embedding(
        #     num_embeddings=4, embedding_dim=self.embedding_dim, max_norm=True
        # )

        # self.projection = nn.Conv1d(
        #     4, self.embedding_dim, kernel_size=3, stride=3, padding=0
        # )

    # nn.LayerNorm(normalized_shape)
    # Rearrange("b e (w) -> b (w) e")
    # Rearrange("b c (w s1) -> b (w) (s1 c)", s1=self.patch_size),
    # nn.Linear(self.patch_size * self.patch_size, self.embedding_dim),

    # self.ConvLayer = nn.Sequential(
    #     nn.Conv1d(4, 32, kernel_size=3, padding="same", stride=1, bias=False),
    #     nn.BatchNorm1d(32),
    #     nn.ReLU(),
    #     nn.Conv1d(32, 64, kernel_size=3, padding="same", stride=1, bias=False),
    #     nn.BatchNorm1d(64),
    #     nn.ReLU(),
    #     nn.Dropout(),
    # )

    # self.position_encoding = PositionalEncoding(
    #     dim=self.embedding_dim, max_len=self.seq_len, dropout=0.1
    # )

    # self.Q = nn.ModuleList(
    #     [
    #         nn.Linear(
    #             in_features=self.embedding_dim, out_features=self.single_head_size
    #         )
    #         for i in range(0, self.multi_head_num)
    #     ]
    # )
    # self.K = nn.ModuleList(
    #     [
    #         nn.Linear(
    #             in_features=self.embedding_dim, out_features=self.single_head_size
    #         )
    #         for i in range(0, self.multi_head_num)
    #     ]
    # )
    # self.V = nn.ModuleList(
    #     [
    #         nn.Linear(
    #             in_features=self.embedding_dim, out_features=self.single_head_size
    #         )
    #         for i in range(0, self.multi_head_num)
    #     ]
    # )

    # self.relu = nn.ModuleList([nn.ReLU() for i in range(0, self.multi_head_num)])
    # self.MultiHeadLinear = nn.Sequential(
    #     nn.LayerNorm(self.single_head_size * self.multi_head_num),
    #     nn.Linear(
    #         in_features=self.single_head_size * self.multi_head_num,
    #         out_features=self.multi_head_size,
    #     ),
    #     nn.ReLU(),
    #     nn.Dropout(p=0.2),
    # )
    # # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
    # # self.flattening = Flattening()
    # self.predictor = nn.Sequential(
    #     nn.BatchNorm1d(1617),
    #     nn.Linear(in_features=1617, out_features=512),
    #     nn.ReLU(),
    #     nn.Dropout(),
    #     nn.BatchNorm1d(512),
    #     nn.Linear(in_features=512, out_features=32),
    #     nn.ReLU(),
    #     nn.BatchNorm1d(32),
    #     nn.Linear(in_features=32, out_features=1),
    # )

    # def attention(self, query, key, value, mask=None, dropout=0.0):
    #     # based on: https://nlp.seas.harvard.edu/2018/04/03/attention.html
    #     d_k = query.size(-1)
    #     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    #     p_attn = F.softmax(scores, dim=-1)
    #     p_attn = F.dropout(p_attn, p=dropout, training=True)
    #     return torch.matmul(p_attn, value), p_attn

    # def forward(self, inputs):

    #     PatchEmbedding()

    # identity = inputs.clone()

    # Case 1 : embedding (256, 33, 64)-> patch
    # input_embd = self.embedding_layer(inputs) * math.sqrt(self.embedding_dim)
    # PatchEmbedding(in_channels = self.embedding_dim, patch_size = , emb_size)
    # Case 2 : one-hot encoding (256, 33, 4) -> patch

    ##############################################################

    # inputs_sums = self.position_encoding(inputs_embs)
    # output = inputs_sums

    # CNN
    #
    # output = encoding.transpose(1, 2)
    # output = self.ConvLayer(output)
    # output = output.transpose(2, 1)  # 256, 33, 64

    ##########################################################################
    # pe_layer = PositionalEncoding(self.embedding_dim)
    # fig, ax = plt.subplots(figsize=(15, 9))
    # cax = ax.matshow(pe_layer.pe.squeeze(0), cmap=plt.cm.YlOrRd)
    # fig.colorbar(cax)
    # ax.set_title("Positional Emcoder Matrix", fontsize=18)
    # ax.set_xlabel("Embedding Dimension", fontsize=14)
    # ax.set_ylabel("Sequence Length", fontsize=14)
    ##########################################################################

    # pAttn_concat = torch.Tensor([]).to(inputs.device)
    # attn_concat = torch.Tensor([]).to(inputs.device)
    # for i in range(0, self.multi_head_num):
    #     query = self.Q[i](output)
    #     key = self.K[i](output)
    #     value = self.V[i](output)
    #     attnOut, p_attn = self.attention(query, key, value, dropout=0.0)
    #     attnOut = self.relu[i](attnOut)
    #     attn_concat = torch.cat((attn_concat, attnOut), dim=2)

    # attn_out = self.MultiHeadLinear(attn_concat)
    # attn_out = self.maxpool(attn_out)
    # attn_out = self.flattening(attn_out)

    # output = self.predictor(attn_out)

    # return output.squeeze()
