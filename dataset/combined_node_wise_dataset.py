import json
import logging
from typing import List

import numpy as np
from torch.utils.data import Dataset


class CombinedNodeWiseDataset(Dataset):
    """
    A dataset that combines node-wise samples from multiple traffic datasets.

    Each dataset (e.g. PEMS03, PEMS04, PEMS07, PEMS08) is loaded independently,
    split by mode (train/valid/test), and then all node-wise samples are
    concatenated into a single flat index space.

    Each sample is shape [L, 1, C] — a single node's time series.

    The total length is:
        sum over all datasets of: time_samples_i * num_nodes_i
    """

    def __init__(self, dataset_names: List[str], train_val_test_ratio: List[float],
                 mode: str, input_len: int, output_len: int,
                 overlap: bool = False, logger: logging.Logger = None) -> None:
        assert mode in ['train', 'valid', 'test'], \
            f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."

        self.dataset_names = dataset_names
        self.train_val_test_ratio = train_val_test_ratio
        self.mode = mode
        self.input_len = input_len
        self.output_len = output_len
        self.overlap = overlap
        self.logger = logger

        # Load all datasets
        self.datasets_data = []     # list of np.ndarray [T_i, N_i, C]
        self.datasets_num_nodes = []
        self.datasets_time_samples = []
        self.cumulative_lengths = [0]  # for index mapping

        for name in dataset_names:
            desc = self._load_description(name)
            data = self._load_data(name, desc)
            num_nodes = desc['num_nodes']
            time_samples = len(data) - input_len - output_len + 1

            self.datasets_data.append(data)
            self.datasets_num_nodes.append(num_nodes)
            self.datasets_time_samples.append(time_samples)

            dataset_len = time_samples * num_nodes
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + dataset_len)

            log_msg = f'[CombinedNodeWise] {name} ({mode}): T={len(data)}, N={num_nodes}, ' \
                      f'time_samples={time_samples}, total_node_samples={dataset_len}'
            if logger:
                logger.info(log_msg)
            else:
                print(log_msg)

        self._total_len = self.cumulative_lengths[-1]
        log_msg = f'[CombinedNodeWise] Total {mode} samples: {self._total_len}'
        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)

    def _load_description(self, dataset_name: str) -> dict:
        path = f'datasets/{dataset_name}/desc.json'
        with open(path, 'r') as f:
            return json.load(f)

    def _load_data(self, dataset_name: str, desc: dict) -> np.ndarray:
        """Load and split data by mode. Returns [T_split, N, C]."""
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
        Map a flat index to (dataset_idx, time_idx, node_idx) and return
        a single-node sample of shape [L, 1, C].
        """
        # Find which dataset this index belongs to
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
