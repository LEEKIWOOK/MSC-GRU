import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math


class AttentionConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
    ):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert (
            self.out_channels % self.groups == 0
        ), "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel = nn.Parameter(
            torch.randn(1, self.out_channels // 1, 1, self.kernel_size),
            requires_grad=True,
        )

        self.key_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride)

        k_out = k_out.contiguous().view(
            batch, self.groups, self.out_channels // self.groups, width, -1
        )
        v_out = v_out.contiguous().view(
            batch, self.groups, self.out_channels // self.groups, width, -1
        )

        q_out = q_out.view(
            batch, self.groups, self.out_channels // self.groups, width, 1
        )

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum("bncwk,bncwk -> bncw", out, v_out).view(batch, -1, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(
            self.value_conv.weight, mode="fan_out", nonlinearity="relu"
        )
        init.kaiming_normal_(
            self.query_conv.weight, mode="fan_out", nonlinearity="relu"
        )

        init.normal_(self.rel, 0, 1)


# class AttentionStem(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride=1,
#         padding=0,
#         groups=1,
#         m=4,
#         bias=False,
#     ):
#         super(AttentionStem, self).__init__()
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.groups = groups
#         self.m = m

#         assert (
#             self.out_channels % self.groups == 0
#         ), "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

#         self.emb_a = nn.Parameter(
#             torch.randn(out_channels // groups, kernel_size), requires_grad=True
#         )
#         self.emb_b = nn.Parameter(
#             torch.randn(out_channels // groups, kernel_size), requires_grad=True
#         )
#         self.emb_mix = nn.Parameter(
#             torch.randn(m, out_channels // groups), requires_grad=True
#         )

#         self.key_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
#         self.query_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
#         self.value_conv = nn.ModuleList(
#             [
#                 nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
#                 for _ in range(m)
#             ]
#         )

#         self.reset_parameters()

#     def forward(self, x):
#         batch, channels, width = x.size()

#         padded_x = F.pad(x, [self.padding, self.padding])

#         q_out = self.query_conv(x)
#         k_out = self.key_conv(padded_x)
#         v_out = torch.stack(
#             [self.value_conv[_](padded_x) for _ in range(self.m)], dim=0
#         )

#         k_out = k_out.unfold(2, self.kernel_size, self.stride)
#         v_out = v_out.unfold(3, self.kernel_size, self.stride)

#         k_out = k_out[:, :, :width, :, :]
#         v_out = v_out[:, :, :, :width, :, :]

#         emb_logit_a = torch.einsum("mc,ca->ma", self.emb_mix, self.emb_a)
#         emb_logit_b = torch.einsum("mc,cb->mb", self.emb_mix, self.emb_b)
#         emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
#         emb = F.softmax(emb.view(self.m, -1), dim=0).view(
#             self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size
#         )

#         v_out = emb * v_out

#         k_out = k_out.contiguous().view(
#             batch, self.groups, self.out_channels // self.groups, width, -1
#         )
#         v_out = v_out.contiguous().view(
#             self.m,
#             batch,
#             self.groups,
#             self.out_channels // self.groups,
#             width,
#             -1,
#         )
#         v_out = torch.sum(v_out, dim=0).view(
#             batch, self.groups, self.out_channels // self.groups, width, -1
#         )

#         q_out = q_out.view(
#             batch, self.groups, self.out_channels // self.groups, width, 1
#         )

#         out = q_out * k_out
#         out = F.softmax(out, dim=-1)
#         out = torch.einsum("bnchwk,bnchwk->bnchw", out, v_out).view(batch, -1, width)

#         return out

#     def reset_parameters(self):
#         init.kaiming_normal_(self.key_conv.weight, mode="fan_out", nonlinearity="relu")
#         init.kaiming_normal_(
#             self.query_conv.weight, mode="fan_out", nonlinearity="relu"
#         )
#         for _ in self.value_conv:
#             init.kaiming_normal_(_.weight, mode="fan_out", nonlinearity="relu")

#         init.normal_(self.emb_a, 0, 1)
#         init.normal_(self.emb_b, 0, 1)
#         init.normal_(self.emb_mix, 0, 1)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channels, out_channels, stride=1, groups=1, base_width=64):
        super(Bottleneck, self).__init__()
        self.stride = stride
        width = int(out_channels * (base_width / 64.0)) * groups

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.conv2 = nn.Sequential(
            AttentionConv(width, width, kernel_size=7, padding=3, groups=8),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(width, self.expansion * out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.expansion * out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(self.expansion * out_channels),
            )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.stride >= 2:
            out = F.avg_pool1d(out, self.stride)

        out += self.shortcut(x)
        out = F.relu(out)
        out = self.dropout(out)

        return out


class Model(nn.Module):
    def __init__(self, block, num_blocks, stem=False):
        super(Model, self).__init__()
        self.in_places = 16

        if stem:
            self.init = nn.Sequential(
                # CIFAR10
                AttentionStem(
                    in_channels=4,
                    out_channels=16,
                    kernel_size=4,
                    stride=1,
                    padding=2,
                    groups=1,
                ),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                # For ImageNet
                # AttentionStem(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2, groups=1),
                # nn.BatchNorm2d(64),
                # nn.ReLU(),
                # nn.MaxPool2d(4, 4)
            )
        else:
            self.init = nn.Sequential(
                # CIFAR10
                nn.Conv1d(4, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                # For ImageNet
                # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                # nn.BatchNorm2d(64),
                # nn.ReLU(),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=1)  # 2 -> 1
        self.dense = nn.Sequential(
            # nn.BatchNorm1d(1024),
            # nn.Linear(in_features=1024, out_features=256),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.BatchNorm1d(256),
            nn.Linear(in_features=256, out_features=16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(in_features=16, out_features=1),
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_places, planes, stride))
            self.in_places = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.one_hot(x).to(torch.float)
        out = out.transpose(1, 2)

        out = self.init(out)  # 256, 16, 33
        out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool1d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dense(out)

        return out.squeeze()


def ResNet(stem=False):
    return Model(Bottleneck, [2, 2], stem=stem)


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters
