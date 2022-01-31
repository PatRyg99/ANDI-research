from typing import Tuple
from dataclasses import dataclass


@dataclass
class TrainClassifierConfig:
    seed: int = 0

    dataset_path: str = ".data/datasets/mily-data/Andi_dane"
    one_hotted: bool = True

    epochs: int = 100
    bs: int = 1024
    lr: float = 0.00063

    dim: int = 2
    classes: int = 5
    channels: Tuple[int] = (64, 128, 256, 512)
    stride: int = 1
    main_kernel_size: int = 1
    branch_kernel_sizes: Tuple[int] = (3, 5)
