from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.AttentionBlock import AttentionBlock
from torchinfo import summary
class SingleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 padding: Union[str, int], dilation: int):
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.single_conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            SingleConv(in_channels, out_channels, kernel_size, "same", dilation),
            SingleConv(out_channels, out_channels, kernel_size, "same", dilation),
        )

    def forward(self, x):
        out = self.conv_block(x)
        return out

class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor, kernel_size, dilation):
        super(Decoder, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="linear")
        self.single_conv = SingleConv(in_channels, out_channels, 1, "same", dilation)
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, dilation)
        self.attention_block = AttentionBlock(out_channels, out_channels, out_channels)

    def forward(self, x1, x2) -> torch.Tensor:
        x1 = self.up(x1)
        x1 = self.single_conv(x1)
        # print(x1.shape)
        psi = self.attention_block(x1, x2)
        assert x1.shape[-1] == x2.shape[-1]
        x = torch.cat([x2 * psi, x1], dim=1)

        return self.conv_block(x)



class BLAST(nn.Module):
    def __init__(self):
        super(TANet, self).__init__()
        self.feature_activation = None
        self.bn = torch.nn.BatchNorm1d(2)
        kernel_size = 7
        self.encoders = nn.ModuleList(
            [
                ConvBlock(2, 32, kernel_size, 1),
                ConvBlock(32, 64, kernel_size, 1),
                ConvBlock(64, 128, kernel_size, 1)
            ]
        )

        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.pool = nn.AvgPool1d(2)
        self.attention_maps = []  # List of attention maps per decoder
        self.lstm1 = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )


        self.decoders = nn.ModuleList(
            [
                Decoder(256, 128, 2, kernel_size, 1),
                Decoder(128, 64, 2, kernel_size, 1),
                Decoder(64, 32, 2, kernel_size, 1)
            ]
        )

        self.dense = nn.Conv1d(32, 2, kernel_size=1, padding='same')


    def freeze(self, module):
        for name, para in module.named_parameters():
            para.requires_grad_(False)

    @torch.no_grad()
    def initialize(self):
        self.apply(self.initialize_weights)

    @torch.no_grad()
    def initialize_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.bn(x)
        features_enc = []
        for idx, enc in enumerate(self.encoders):
            x = enc(x)
            features_enc.append(x)
            if idx >= 2:
                x = self.dropout_5(x)
            else:
                x = self.dropout_2(x)
            x = self.pool(x)

        # print(x.shape)
        x = x.permute(0, 2, 1)
        x, (hn, cn) = self.lstm1(x)
        x = self.dropout_5(x)
        x, (hn, cn) = self.lstm2(x)

        self.feature_activation = x.detach().cpu()  #
        x = x.permute(0, 2, 1)

        # reverse the encoder outputs (inplace)
        features_enc.reverse()
        # print(x.shape)
        # calculate output of all decoders, using the encoder outputs as skip connections
        self.attention_maps = []  # List of attention maps per decoder

        for idx, (dec, x_enc) in enumerate(zip(self.decoders, features_enc)):
            if idx >= 1:
                x = self.dropout_2(x)
            else:
                x = self.dropout_5(x)
            x = dec(x, x_enc)
            # print(x.shape)
        self.attention_maps.append(dec.attention_block.att_map[0].squeeze(0))
        logits = self.dense(x)
        # print(logits.shape)
        return logits



if __name__ == "__main__":
    data = torch.randn(32, 2, 20 *  100)
    model = BLAST()
    # summary(model, input_size=(32, 2, 20 *  100))
    output=model(data)
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {output.shape}")
