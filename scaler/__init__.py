import json
from typing import List

import numpy as np
import torch

from basicts.scaler import ZScoreScaler


class CombinedZScoreScaler(ZScoreScaler):
    """
    ZScoreScaler that computes mean/std across the training portions
    of multiple datasets combined.

    All datasets are assumed to share the same feature layout (channel 0 = target).
    The global mean and std are computed over all training data from all datasets.
    """

    def __init__(self, dataset_names: List[str], train_ratio: float,
                 norm_each_channel: bool, rescale: bool):
        # Skip the parent __init__ to avoid loading a single dataset.
        # Instead, manually set the dataclass fields and compute stats.
        # We call object.__init__ and set fields directly.
        object.__init__(self)
        self.dataset_name = '+'.join(dataset_names)
        self.train_ratio = train_ratio
        self.norm_each_channel = norm_each_channel
        self.rescale = rescale
        self.target_channel = 0

        # Collect all training data from all datasets
        all_train_data = []
        for name in dataset_names:
            desc_path = f'datasets/{name}/desc.json'
            with open(desc_path, 'r') as f:
                desc = json.load(f)
            data_path = f'datasets/{name}/data.dat'
            data = np.memmap(data_path, dtype='float32', mode='r', shape=tuple(desc['shape']))
            train_size = int(len(data) * train_ratio)
            # shape: [train_size, num_nodes, C] -> take target channel -> flatten
            train_data = data[:train_size, :, self.target_channel].copy()  # [T, N]
            all_train_data.append(train_data.reshape(-1))  # flatten to 1D

        combined = np.concatenate(all_train_data)

        # For combined datasets with different num_nodes, per-channel norm doesn't apply
        # We always do global norm (since nodes are mixed across datasets)
        self.mean = np.mean(combined).astype(np.float32)
        self.std = np.std(combined).astype(np.float32)
        if self.std == 0:
            self.std = np.float32(1.0)

        self.mean = torch.tensor(self.mean)
        self.std = torch.tensor(self.std)
