import os

from tqdm import tqdm

import torch
import numpy as np
from monai.transforms import Compose

from src.transforms import ToFloatTensord, Normalized
from src.classifier import TrajectoryClassifier
from configs.inference_config import InferenceClassifierConfig


def remap_preds(labels, preds):
    unique_labels = np.unique(labels)
    unique_preds = np.unique(preds)

    for i, unique in enumerate(unique_preds):
        preds[preds == unique] = unique_labels[i]

    return preds


def main():
    config = InferenceClassifierConfig()
    os.makedirs(config.out_path, exist_ok=True)

    # Load model
    model = TrajectoryClassifier.load_from_checkpoint(config.checkpoint_path)
    model.cuda()
    model.eval()
    model.freeze()

    # Preprocessing
    preprocess = Compose([Normalized(keys=["input"]), ToFloatTensord(keys=["input"])])

    data_suffixes = ["train", "val", "test"]

    for suffix in data_suffixes:

        # Load data
        X = np.load(os.path.join(config.dataset_path, f"X_{suffix}.npy"))
        y = np.load(os.path.join(config.dataset_path, f"y_{suffix}.npy"))

        if config.one_hotted:
            y = np.argmax(y, axis=1)

        X = preprocess([{"input": x} for x in X])
        X = torch.stack([x["input"] for x in X]).cuda()
        bs = config.bs

        preds = np.zeros((len(X), config.classes))

        with tqdm(total=len(X) // bs + 1) as pbar:
            pbar.set_description(suffix)

            for batch_start in range(0, len(X), bs):

                batch_data = X[batch_start : batch_start + bs]
                pred = model(batch_data).detach().cpu().numpy()
                preds[batch_start : batch_start + bs] = pred

                pbar.update()

        # De-one-hot and remap
        preds = np.argmax(preds, axis=1).astype(int)

        # Save prediction
        np.save(os.path.join(config.out_path, f"predictions_{suffix}.npy"), preds)


if __name__ == "__main__":
    main()
