from typing import Tuple

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block wrapper"""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x) + x


class MultiBranchBlock(nn.Module):
    """Multi branch block wrapper"""

    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list

    def forward(self, x):
        return torch.cat([module(x) for module in self.module_list], dim=1)


class TrajectoryNet(nn.Module):
    """TrajectoryNet based on PointNet architecture"""

    def __init__(
        self,
        dim: int,
        channels: Tuple[int],
        classes: int,
        stride: int,
        main_kernel_size: int,
        branch_kernel_sizes: Tuple[int],
    ):
        super().__init__()

        self.dim = dim
        self.classes = classes

        channels = (dim,) + channels

        self.stem = nn.Sequential(
            *[
                self.block(
                    channels[i],
                    channels[i + 1],
                    stride,
                    main_kernel_size,
                    branch_kernel_sizes,
                )
                for i in range(len(channels) - 2)
            ],
            nn.Conv1d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=main_kernel_size,
                padding_mode="replicate",
                bias=True,
            ),
        )

        self.head = nn.Sequential(
            nn.Linear(channels[-1], 256, bias=True),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, self.classes, bias=True),
        )

    def block(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        main_kernel_size: int,
        branch_kernel_sizes: Tuple[int],
    ):
        branch_channels = out_channels // len(branch_kernel_sizes)

        return nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=main_kernel_size,
                stride=stride,
                padding_mode="replicate",
                bias=True,
            ),
            MultiBranchBlock(
                nn.ModuleList(
                    [
                        ResBlock(
                            nn.Sequential(
                                nn.Conv1d(
                                    in_channels=branch_channels,
                                    out_channels=branch_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=(kernel_size - 1) // 2,
                                    padding_mode="replicate",
                                    bias=True,
                                ),
                                nn.BatchNorm1d(branch_channels),
                                nn.GELU(),
                            )
                        )
                        for kernel_size in branch_kernel_sizes
                    ]
                )
            ),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        bs = x.shape[0]
        x = x.reshape(bs, self.dim, -1)

        output = self.stem(x)
        output = output.max(dim=2)[0]
        output = self.head(output)

        return output
