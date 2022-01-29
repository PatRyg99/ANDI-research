from typing import List, Union, Tuple
from dataclasses import dataclass, field


@dataclass
class GenerateDataConfig:
    N: int = 5000
    T: int = 512
    exponents: Union[float, List[float]] = 0.7
    models: Union[bool, int, List[int]] = field(default_factory=lambda: [0, 1, 2, 4])
    dimension: int = 2

    train_size: int = 0.7
    out_path: str = ".data/datasets/alpha=0.7/"
