from dataclasses import dataclass


@dataclass
class InferenceClassifierConfig:
    seed: int = 0
    checkpoint_path: str = "/home/patryk/Desktop/andi/.data/checkpoints/epoch=26-step=1295.ckpt"
    out_path: str = "/home/patryk/Desktop/andi/.data/predictions/mily-kernel=5-noise"

    dataset_path: str = ".data/datasets/mily-data/Andi_dane"
    one_hotted: bool = True
    dim: int = 2
    classes: int = 5

    bs: int = 1024
