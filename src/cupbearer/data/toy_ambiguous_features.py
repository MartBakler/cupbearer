from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset

from ._shared import DatasetConfig


@dataclass
class ToyFeaturesConfig(DatasetConfig):
    correlated: bool = True
    size: int = 1000
    noise: float = 0.1
    num_classes: int = 2

    def _build(self):
        return ToyDataset(self.size, self.correlated, self.noise)


class ToyDataset(Dataset):
    def __init__(self, size: int, correlated: bool, noise: float):
        self.size = size
        self.correlated = correlated
        self.noise = noise
        self._generate_data()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def _generate_data(self):
        x = np.random.randn(self.size)
        labels = (x > 0).astype(int)
        feature1 = x + np.random.randn(self.size) * self.noise

        if self.correlated:
            feature2 = x + np.random.randn(self.size) * self.noise
        else:
            feature2 = -x + np.random.randn(self.size) * self.noise

        self.features = np.stack((feature1, feature2), axis=1)
        self.labels = labels
