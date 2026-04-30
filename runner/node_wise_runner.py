from typing import Dict

import torch

from basicts.runners import SimpleTimeSeriesForecastingRunner


class NodeWiseTimeSeriesForecastingRunner(SimpleTimeSeriesForecastingRunner):
    """
    A runner for node-wise time series forecasting.

    This runner handles data of shape [B, L, 1, C] where each sample
    corresponds to a single node. It extends SimpleTimeSeriesForecastingRunner
    with minimal changes:
      - The forward pass works with num_nodes=1 per sample.
      - Scaler, metrics, and evaluation all work element-wise so
        per-node processing produces equivalent results to full-graph.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        # Store the actual num_nodes for reference (used in test aggregation)
        self.actual_num_nodes = cfg['MODEL'].get('ACTUAL_NUM_NODES', None)

    def forward(self, data: Dict, epoch: int = None, iter_num: int = None,
                train: bool = True, **kwargs) -> Dict:
        """
        Forward pass for node-wise data.

        Data shape: inputs/target are [B, L, 1, C] (single node per sample).
        """

        data = self.preprocessing(data)

        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)  # [B, L, 1, C]
        future_data = self.to_running_device(future_data)    # [B, L, 1, C]
        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Forward pass through the model
        model_return = self.model(
            history_data=history_data,
            future_data=future_data_4_dec,
            batch_seen=iter_num,
            epoch=epoch,
            train=train
        )

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        # Ensure output shape: [B, L, 1, C]
        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            f"Output shape mismatch. Expected [{batch_size}, {length}, {num_nodes}], " \
            f"got {list(model_return['prediction'].shape)[:3]}."

        model_return = self.postprocessing(model_return)

        return model_return
