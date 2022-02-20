from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class ST_Attn(nn.Module):
    def __init__(self, in_dim):
        super(ST_Attn, self).__init__()
        self.chanel_in = in_dim
        chanel_out = in_dim // 8

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=chanel_out, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=chanel_out, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channel, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # B C N
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)  # B N N
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # B C N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out, attention


class STNN(nn.Module):
    def __init__(self, num_features, num_timesteps_input, num_timesteps_output, node_num,
                 dropout=0.2):
        super(STNN, self).__init__()

        self.idx = 0
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output

        f = 3
        # self.io_channels = [num_features, 64, 128, 256]
        self.blocks = 2
        self.io_channels = [num_features, 32, 64]
        assert len(self.io_channels) == self.blocks + 1

        self.init_convs = nn.ModuleList()
        self.st_region_convs = nn.ModuleList()
        self.st_pair_t_convs = nn.ModuleList()
        self.st_pair_n_convs = nn.ModuleList()
        self.st_condense = nn.ModuleList()
        self.st_attns = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.denses = nn.ModuleList()

        for i in range(self.blocks):
            assert i < 6
            # kernel size = (height dimension, width dimension)
            self.init_convs.append(nn.Conv2d(in_channels=self.io_channels[i],
                                             out_channels=self.io_channels[i + 1],
                                             kernel_size=(1, 1)))
            self.st_region_convs.append(nn.Conv2d(in_channels=self.io_channels[i + 1],
                                                  out_channels=self.io_channels[i + 1],
                                                  kernel_size=(f, f),  # (i*2+3, i*2+3), (3, 3)
                                                  padding=(1, 1)))  # (i+1, i+1), (1, 1)
            self.st_pair_t_convs.append(nn.Conv2d(in_channels=self.io_channels[i + 1],
                                                  out_channels=self.io_channels[i + 1],
                                                  kernel_size=(1, f),
                                                  padding=(0, 1))),
            self.st_pair_n_convs.append(nn.Conv2d(in_channels=self.io_channels[i + 1],
                                                  out_channels=self.io_channels[i + 1],
                                                  kernel_size=(f, 1),
                                                  padding=(1, 0)))
            self.st_condense.append(nn.Conv2d(in_channels=self.io_channels[i + 1] * 3,
                                              out_channels=self.io_channels[i + 1],
                                              kernel_size=(1, 1),
                                              padding=(0, 0)))
            self.st_attns.append(ST_Attn(self.io_channels[i + 1]))

            self.residuals.append(nn.Conv2d(in_channels=self.io_channels[i + 1],
                                            out_channels=self.io_channels[i + 1],
                                            kernel_size=(1, 1),
                                            padding=(0, 0)))

            self.denses.append(
                nn.Linear((self.num_timesteps_input) * self.io_channels[i + 1] * node_num,
                          self.num_timesteps_output))

        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, input):
        # Convert NHWC to NCHW. NCHW is better optimized for cuDNN.
        input = input.float()
        input = input.permute(0, 3, 1, 2).contiguous()
        x = input

        # ST-Conv Module
        for i in range(self.blocks):
            x = self.init_convs[i](x)
            input_x = x

            x, att = self.st_attns[i](x)

            region = F.leaky_relu(self.st_region_convs[i](x), 0.2)
            pair_t = F.leaky_relu(self.st_pair_t_convs[i](x), 0.2)
            pair_n = F.leaky_relu(self.st_pair_n_convs[i](x), 0.2)
            x = torch.cat([region, pair_t, pair_n], dim=1)
            x = F.leaky_relu(self.st_condense[i](x), 0.2)
            x = F.leaky_relu(x + self.residuals[i](input_x), 0.2)

        x = self.dropout_layer(x)

        # NCHW to NHWC
        x = x.permute(0, 2, 3, 1)

        x = self.denses[self.blocks - 1](x.reshape((x.shape[0], -1)))

        return x
