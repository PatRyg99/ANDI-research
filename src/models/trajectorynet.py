from typing import List
import torch.nn as nn


class TrajectoryNet(nn.Module):
    def __init__(
        self,
        dim: int,
        channels: List[int],
        classes: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()

        self.dim = dim
        self.classes = classes

        channels = (dim,) + channels

        self.stem = nn.Sequential(
            *[
                self.block(channels[i], channels[i + 1], kernel_size, stride)
                for i in range(len(channels) - 2)
            ],
            nn.Conv1d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=kernel_size,
                bias=True,
            ),
        )

        self.head = nn.Sequential(
            nn.Linear(channels[-1], 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.classes, bias=True),
        )

    def block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        return nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=True,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        bs = x.shape[0]
        x = x.reshape(bs, self.dim, -1)

        output = self.stem(x)
        output = output.max(dim=2)[0]
        output = self.head(output)

        return output
