from typing import Dict

import torch

from basicts.runners import SimpleTimeSeriesForecastingRunner


class GraphConditionedRunner(SimpleTimeSeriesForecastingRunner):
    """
    Runner for the final NodeSTIDGraphConditioned model.

    This runner:
    1. Uses standard [B, L, N, C] data (full graph, not node-wise).
    2. Loads pretrained NodeSTIDv2 weights into the backbone at initialization.
    3. The rest of training/validation/test follows the standard flow.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        # Load pretrained backbone if path is specified
        pretrained_path = cfg['MODEL'].get('PRETRAINED_BACKBONE_PATH', None)
        freeze_backbone = cfg['MODEL'].get('FREEZE_BACKBONE', True)

        if pretrained_path is not None:
            self.model.load_pretrained_backbone(pretrained_path, strict=False)
        else:
            self.logger.warning(
                "[GraphConditionedRunner] No PRETRAINED_BACKBONE_PATH specified. "
                "Training from scratch without pretrained weights."
            )
