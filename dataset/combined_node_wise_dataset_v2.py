import json
import logging
from typing import List

import numpy as np
from torch.utils.data import Dataset


class CombinedNodeWiseDatasetV2(Dataset):
    """
    Combined node-wise dataset that appends a dataset ID as an extra channel.

    Each sample has shape [L, 1, C+1] where the last channel is the dataset ID
    (integer 0, 1, 2, ... num_datasets-1).

    Original data channels: [flow, time_of_day, day_of_week]
    After augmentation:     [flow, time_of_day, day_of_week, dataset_id]

    Train/val/test each contain samples from ALL datasets (each dataset is
    independently split by the train_val_test_ratio).
    """

    def __init__(self, dataset_names: List[str], train_val_test_ratio: List[float],
                 mode: str, input_len: int, output_len: int,
                 overlap: bool = False, logger: logging.Logger = None) -> None:
        assert mode in ['train', 'valid', 'test'], \
            f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."

        self.dataset_names = dataset_names
        self.num_datasets = len(dataset_names)
        self.train_val_test_ratio = train_val_test_ratio
        self.mode = mode
        self.input_len = input_len
        self.output_len = output_len
        self.overlap = overlap
        self.logger = logger

        # Load all datasets
        self.datasets_data = []         # list of np.ndarray [T_i, N_i, C+1]
        self.datasets_num_nodes = []
        self.datasets_time_samples = []
        self.cumulative_lengths = [0]

        for ds_idx, name in enumerate(dataset_names):
            desc = self._load_description(name)
            raw_data = self._load_data(name, desc)  # [T, N, C]
            num_nodes = desc['num_nodes']

            # Append dataset_id as the last channel: [T, N, C+1]
            T, N, C = raw_data.shape
            dataset_id_channel = np.full((T, N, 1), fill_value=ds_idx, dtype=np.float32)
            data = np.concatenate([raw_data, dataset_id_channel], axis=-1)  # [T, N, C+1]

            time_samples = len(data) - input_len - output_len + 1

            self.datasets_data.append(data)
            self.datasets_num_nodes.append(num_nodes)
            self.datasets_time_samples.append(time_samples)

            dataset_len = time_samples * num_nodes
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + dataset_len)

            log_msg = (f'[CombinedNodeWiseV2] {name} (id={ds_idx}, {mode}): '
                       f'T={len(data)}, N={num_nodes}, '
                       f'time_samples={time_samples}, total_node_samples={dataset_len}')
            if logger:
                logger.info(log_msg)
            else:
                print(log_msg)

        self._total_len = self.cumulative_lengths[-1]
        log_msg = f'[CombinedNodeWiseV2] Total {mode} samples: {self._total_len}'
        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)

    def _load_description(self, dataset_name: str) -> dict:
        path = f'datasets/{dataset_name}/desc.json'
        with open(path, 'r') as f:
            return json.load(f)

    def _load_data(self, dataset_name: str, desc: dict) -> np.ndarray:
        """Load and split raw data by mode. Returns [T_split, N, C] (without dataset_id yet)."""
        data_path = f'datasets/{dataset_name}/data.dat'
        data = np.memmap(data_path, dtype='float32', mode='r', shape=tuple(desc['shape']))

        total_len = len(data)
        valid_len = int(total_len * self.train_val_test_ratio[1])
        test_len = int(total_len * self.train_val_test_ratio[2])
        train_len = total_len - valid_len - test_len

        minimal_len = self.input_len + self.output_len
        overlap = self.overlap
        split_len = {'train': train_len, 'valid': valid_len, 'test': test_len}[self.mode]
        if minimal_len > split_len:
            overlap = True
            msg = f'{dataset_name} {self.mode} set is too short, enabling overlap.'
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)

        if self.mode == 'train':
            offset = self.output_len if overlap else 0
            return data[:train_len + offset].copy()
        elif self.mode == 'valid':
            offset_left = self.input_len - 1 if overlap else 0
            offset_right = self.output_len if overlap else 0
            return data[train_len - offset_left: train_len + valid_len + offset_right].copy()
        else:  # test
            offset = self.input_len - 1 if overlap else 0
            return data[train_len + valid_len - offset:].copy()

    def __getitem__(self, index: int) -> dict:
        """
        Map flat index -> (dataset_idx, time_idx, node_idx).

        Returns:
            dict with:
                'inputs': [input_len, 1, C+1]  (last channel = dataset_id)
                'target': [output_len, 1, C+1]  (last channel = dataset_id)
        """
        # Find which dataset
        dataset_idx = 0
        for i in range(len(self.datasets_data)):
            if index < self.cumulative_lengths[i + 1]:
                dataset_idx = i
                break

        local_index = index - self.cumulative_lengths[dataset_idx]
        num_nodes = self.datasets_num_nodes[dataset_idx]
        time_idx = local_index // num_nodes
        node_idx = local_index % num_nodes

        data = self.datasets_data[dataset_idx]
        history = data[time_idx:time_idx + self.input_len, node_idx:node_idx + 1, :]
        future = data[time_idx + self.input_len:time_idx + self.input_len + self.output_len,
                      node_idx:node_idx + 1, :]
        return {'inputs': history, 'target': future}

    def __len__(self) -> int:
        return self._total_len
