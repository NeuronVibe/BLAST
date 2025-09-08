from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .AttentionBlock import AttentionBlock

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
        # 这里或许可以更换
        # TODO

        self.up = nn.Upsample(scale_factor=scale_factor, mode="linear")
        self.single_conv = SingleConv(in_channels, out_channels, 1, "same", dilation)
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, dilation)
        self.attention_block = AttentionBlock(out_channels, out_channels, out_channels)

    def forward(self, x1, x2) -> torch.Tensor:
        # upsample the output data of the previous decoder
        x1 = self.up(x1)
        # feed the upsampled tensor into a single convolution, which halves the number of channels
        x1 = self.single_conv(x1)

        psi = self.attention_block(x1, x2)
        assert x1.shape[-1] == x2.shape[-1]
        # concatenate the output of the corresponding encoder with the result of the `SingleConv` module, which doubles
        # the number of channels again
        x = torch.cat([x2 * psi, x1], dim=1)
        # x = torch.cat([x2 , x1 * psi], dim=1)
        # x = torch.cat([x2, x1], dim=1)

        # feed the concatenated tensor into a double convolution, which halves the number of channels once more
        return self.conv_block(x)

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        kernel_size = 7
        self.encoders = nn.ModuleList(
            [
                ConvBlock(2, 32, kernel_size, 1),
                ConvBlock(32, 64, kernel_size, 1)
            ]
        )

        self.dropout = nn.Dropout(p=0.2)
        # self.dropout2 = nn.Dropout(p=0.2)
        self.maxpool = nn.MaxPool1d(2)

        self.conv_list = nn.Sequential(
            SingleConv(64, 128, kernel_size, "same", 1),
            SingleConv(128, 128, kernel_size, "same", 1)
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            # bidirectional=True,
        )


        self.decoders = nn.ModuleList(
            [
                Decoder(128, 64, 2, kernel_size, 1),
                Decoder(64, 32, 2, kernel_size, 1)
            ]
        )
        # create convolution for dense segmentation using as many channels as classes to be classified
        self.dense = nn.Conv1d(32, 2, kernel_size=1, padding='same')

    def forward(self, x):
        # x = x[:, 1:2, :]
        # save outputs of all encoders except the last one
        features_enc = []
        for enc in self.encoders:
            x = enc(x)
            features_enc.append(x)
            x = self.dropout(x)
            x = self.maxpool(x)
        x = self.conv_list(x)

        x = x.permute(0, 2, 1)
        x, (hn, cn) = self.lstm(x)
        x = x.permute(0, 2, 1)

        # reverse the encoder outputs (inplace)
        features_enc.reverse()

        # calculate output of all decoders, using the encoder outputs as skip connections
        for dec, x_enc in zip(self.decoders, features_enc):
            x = self.dropout(x)
            x = dec(x, x_enc)

        # calculate the dense segmentation using the last convolution
        logits = self.dense(x)
        print(logits.shape)
        print(logits)
        return logits

