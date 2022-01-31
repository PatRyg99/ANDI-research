from dataclasses import dataclass


@dataclass
class InferenceClassifierConfig:
    seed: int = 0
    checkpoint_path: str = "/home/patryk/Desktop/andi/.data/checkpoints/epoch=63-step=3071.ckpt"
    out_path: str = "/home/patryk/Desktop/andi/.data/predictions/mily-multi-branch-L-gelu-noise"

    dataset_path: str = ".data/datasets/mily-data/Andi_dane"
    one_hotted: bool = True
    dim: int = 2
    classes: int = 5

    bs: int = 1024
