import numpy as np
from monai.transforms.transform import MapTransform, Randomizable
from monai.transforms.utility.dictionary import ToTensord
from monai.config import KeysCollection


class ToFloatTensord(ToTensord):
    """Cast numpy to float32 tensor"""

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = self.converter(d[key]).float()
        return d


class Normalized(MapTransform):
    """Normalize to [0, 1]"""

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            sample = d[key]
            sample = np.nan_to_num((sample - sample.min()) / sample.ptp())

            d[key] = sample
        return d


class Standardized(MapTransform):
    """Standarize to mean of 0 and std of 1"""

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            sample = d[key]
            sample = np.nan_to_num((sample - sample.mean()) / sample.std())

            d[key] = sample
        return d


class RandomNormalNoised(Randomizable, MapTransform):
    """Random normal noise"""

    def __init__(
        self, keys: KeysCollection, std: float = 0.01, allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.std = std

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            data = d[key]
            jitter = self.R.normal(0.0, self.std, size=(data.shape))

            data += jitter
            d[key] = data

        return d
