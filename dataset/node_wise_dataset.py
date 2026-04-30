import inspect
import json
import logging
from typing import List

import numpy as np

from basicts.data.base_dataset import BaseDataset


class NodeWiseTimeSeriesForecastingDataset(BaseDataset):
    """
    A dataset that treats each node independently.

    Instead of returning samples of shape [L, N, C] (all nodes at once),
    this dataset returns samples of shape [L, 1, C] (one node at a time).

    The total number of samples is multiplied by the number of nodes:
        total_samples = (T - input_len - output_len + 1) * num_nodes

    Each index maps to a specific (time_step, node) pair.
    """

    def __init__(self, dataset_name: str, train_val_test_ratio: List[float],
                 mode: str, input_len: int, output_len: int,
                 overlap: bool = False, logger: logging.Logger = None) -> None:
        assert mode in ['train', 'valid', 'test'], \
            f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."
        super().__init__(dataset_name, train_val_test_ratio, mode, input_len, output_len, overlap)
        self.logger = logger

        self.data_file_path = f'datasets/{dataset_name}/data.dat'
        self.description_file_path = f'datasets/{dataset_name}/desc.json'
        self.description = self._load_description()
        self.num_nodes = self.description['num_nodes']
        self.data = self._load_data()

    def _load_description(self) -> dict:
        try:
            with open(self.description_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Description file not found: {self.description_file_path}') from e
        except json.JSONDecodeError as e:
            raise ValueError(f'Error decoding JSON file: {self.description_file_path}') from e

    def _load_data(self) -> np.ndarray:
        """Load time series data and split by mode. Data shape: [T, N, C]."""

        try:
            data = np.memmap(self.data_file_path, dtype='float32', mode='r',
                             shape=tuple(self.description['shape']))
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f'Error loading data file: {self.data_file_path}') from e

        total_len = len(data)
        valid_len = int(total_len * self.train_val_test_ratio[1])
        test_len = int(total_len * self.train_val_test_ratio[2])
        train_len = total_len - valid_len - test_len

        # Automatically configure the overlap parameter
        minimal_len = self.input_len + self.output_len
        if minimal_len > {'train': train_len, 'valid': valid_len, 'test': test_len}[self.mode]:
            self.overlap = True
            dataset_label = {'train': 'Training', 'valid': 'Validation', 'test': 'Test'}[self.mode]
            msg = f'{dataset_label} dataset is too short, enabling overlap.'
            if self.logger is not None:
                self.logger.info(msg)
            else:
                print(msg)

        if self.mode == 'train':
            offset = self.output_len if self.overlap else 0
            return data[:train_len + offset].copy()
        elif self.mode == 'valid':
            offset_left = self.input_len - 1 if self.overlap else 0
            offset_right = self.output_len if self.overlap else 0
            return data[train_len - offset_left: train_len + valid_len + offset_right].copy()
        else:  # test
            offset = self.input_len - 1 if self.overlap else 0
            return data[train_len + valid_len - offset:].copy()

    def __getitem__(self, index: int) -> dict:
        """
        Retrieve a single-node sample.

        The index is decomposed into (time_index, node_index):
            time_index = index // num_nodes
            node_index = index % num_nodes

        Returns:
            dict with:
                'inputs': np.ndarray of shape [input_len, 1, C]
                'target': np.ndarray of shape [output_len, 1, C]
        """
        time_idx = index // self.num_nodes
        node_idx = index % self.num_nodes

        # Select single node: [L, 1, C]
        history_data = self.data[time_idx:time_idx + self.input_len, node_idx:node_idx + 1, :]
        future_data = self.data[time_idx + self.input_len:time_idx + self.input_len + self.output_len,
                                node_idx:node_idx + 1, :]
        return {'inputs': history_data, 'target': future_data}

    def __len__(self) -> int:
        """Total samples = time_samples * num_nodes."""
        time_samples = len(self.data) - self.input_len - self.output_len + 1
        return time_samples * self.num_nodes
