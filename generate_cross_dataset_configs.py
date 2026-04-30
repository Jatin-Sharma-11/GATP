#!/usr/bin/env python3
"""Generate cross-dataset NodeSTID configs if they do not already exist."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PAIRS = [
    ("PEMS03", "PEMS04"),
    ("PEMS03", "PEMS07"),
    ("PEMS03", "PEMS08"),
    ("PEMS04", "PEMS07"),
    ("PEMS04", "PEMS08"),
    ("PEMS07", "PEMS08"),
]

DATASETS = ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]

NODES = {
    "PEMS03": 358,
    "PEMS04": 307,
    "PEMS07": 883,
    "PEMS08": 170,
}

PRETRAIN_TEMPLATE = """import os
import sys

from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse

from .arch import NodeSTIDv2
from .dataset import CombinedNodeWiseDatasetV2
from .runner import NodeWiseTimeSeriesForecastingRunner
from .scaler import CombinedZScoreScaler

############################## Hot Parameters ##############################
DATASET_NAMES = {dataset_names}
EXPERIMENT_TAG = '{tag}'

INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
NORM_EACH_CHANNEL = False
RESCALE = True
NULL_VAL = 0.0

MODEL_ARCH = NodeSTIDv2
MODEL_PARAM = {{
    "input_len": INPUT_LEN,
    "input_dim": 3,
    "embed_dim": 32,
    "output_len": OUTPUT_LEN,
    "num_layer": 3,
    "if_T_i_D": True,
    "if_D_i_W": True,
    "if_dataset_id": True,
    "temp_dim_tid": 32,
    "temp_dim_diw": 32,
    "time_of_day_size": 288,
    "day_of_week_size": 7,
    "num_datasets": len(DATASET_NAMES),
    "dataset_id_dim": 32,
}}
NUM_EPOCHS = 30

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = f'NodeSTIDv2: fast cross-dataset pretraining on {{EXPERIMENT_TAG}} (30 epochs)'
CFG.GPU_NUM = 1
CFG.RUNNER = NodeWiseTimeSeriesForecastingRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = f'PEMS_Combined_v2_{{EXPERIMENT_TAG}}'
CFG.DATASET.TYPE = CombinedNodeWiseDatasetV2
CFG.DATASET.PARAM = EasyDict({{
    'dataset_names': DATASET_NAMES,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
}})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = CombinedZScoreScaler
CFG.SCALER.PARAM = EasyDict({{
    'dataset_names': DATASET_NAMES,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
}})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({{
    'MAE': masked_mae,
    'MAPE': masked_mape,
    'RMSE': masked_rmse,
}})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([f'PEMS_Combined_v2_{{EXPERIMENT_TAG}}', str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {{
    "lr": 0.002,
    "weight_decay": 0.0001,
}}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {{
    "milestones": [1, 15, 24],
    "gamma": 0.5
}}
CFG.TRAIN.CLIP_GRAD_PARAM = {{
    'max_norm': 5.0
}}

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 50000
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 50000

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 50000

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 9, 12]
CFG.EVAL.USE_GPU = True
"""

GRAPH_TEMPLATE = """import glob
import os
import sys

import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from .arch import NodeSTIDGraphConditioned
from .runner import GraphConditionedRunner

############################## Hot Parameters ##############################
DATA_NAME = '{target}'
SOURCE_DATASET_NAMES = {dataset_names}
SOURCE_TAG = '{tag}'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

NUM_NODES = {nodes}
adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "doubletransition")

############################## Pretrained Backbone ##############################
def _resolve_default_pretrained_path() -> str:
    ckpt_root = os.path.join(
        'checkpoints',
        'NodeSTIDv2',
        f'PEMS_Combined_v2_{{SOURCE_TAG}}_30_12_12',
    )
    candidates = sorted(
        glob.glob(os.path.join(ckpt_root, '**', 'NodeSTIDv2_best_val_MAE.pt'), recursive=True)
    )
    if candidates:
        return candidates[0]
    return os.path.join(ckpt_root, 'MISSING', 'NodeSTIDv2_best_val_MAE.pt')

PRETRAINED_PATH = os.getenv('NODESTID_PRETRAINED_PATH', _resolve_default_pretrained_path())
FREEZE_BACKBONE = True

############################## Model Configuration ##############################
MODEL_ARCH = NodeSTIDGraphConditioned
MODEL_PARAM = {{
    "input_len": INPUT_LEN,
    "input_dim": 3,
    "embed_dim": 32,
    "output_len": OUTPUT_LEN,
    "num_layer": 3,
    "if_T_i_D": True,
    "if_D_i_W": True,
    "if_dataset_id": True,
    "temp_dim_tid": 32,
    "temp_dim_diw": 32,
    "time_of_day_size": 288,
    "day_of_week_size": 7,
    "num_datasets": len(SOURCE_DATASET_NAMES),
    "dataset_id_dim": 32,
    "num_nodes": NUM_NODES,
    "freeze_backbone": FREEZE_BACKBONE,
    "default_dataset_id": 0,
    "supports": [torch.tensor(i) for i in adj_mx],
    "use_adaptive_adj": True,
    "node_emb_dim": 10,
    "num_graph_layers": 3,
    "gcn_order": 2,
    "gcn_dropout": 0.1,
}}
NUM_EPOCHS = 100

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = f'NodeSTID-GC: transfer from {{SOURCE_TAG}} to {{DATA_NAME}} (100 epochs)'
CFG.GPU_NUM = 1
CFG.RUNNER = GraphConditionedRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({{
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
}})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({{
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
}})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.PRETRAINED_BACKBONE_PATH = PRETRAINED_PATH
CFG.MODEL.FREEZE_BACKBONE = FREEZE_BACKBONE

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({{
    'MAE': masked_mae,
    'MAPE': masked_mape,
    'RMSE': masked_rmse,
}})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([f'{{DATA_NAME}}_from_{{SOURCE_TAG}}', str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {{
    "lr": 0.002,
    "weight_decay": 0.0001,
}}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {{
    "milestones": [1, 50, 80],
    "gamma": 0.5
}}
CFG.TRAIN.CLIP_GRAD_PARAM = {{
    'max_norm': 5.0
}}

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 9, 12]
CFG.EVAL.USE_GPU = True
"""

RUN_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

GPU="${1:-${GPU:-0}}"
PRETRAIN_ONLY="${PRETRAIN_ONLY:-0}"

pairs=(
  "PEMS03 PEMS04"
  "PEMS03 PEMS07"
  "PEMS03 PEMS08"
  "PEMS04 PEMS07"
  "PEMS04 PEMS08"
  "PEMS07 PEMS08"
)

datasets=(PEMS03 PEMS04 PEMS07 PEMS08)

run_train() {
  local cfg="$1"
  echo "[RUN] python experiments/train.py -c ${cfg} -g ${GPU}"
  python experiments/train.py -c "${cfg}" -g "${GPU}"
}

for pair in "${pairs[@]}"; do
  src_a=$(echo "$pair" | awk '{print $1}')
  src_b=$(echo "$pair" | awk '{print $2}')
  source_tag="${src_a}_${src_b}"

  pre_cfg="baselines/NodeSTID/PEMS_Combined_v2_${source_tag}_E30.py"
  if [[ ! -f "$pre_cfg" ]]; then
    echo "[SKIP] Missing pretrain config: ${pre_cfg}"
    continue
  fi

  run_train "$pre_cfg"

  if [[ "$PRETRAIN_ONLY" == "1" ]]; then
    continue
  fi

  ckpt_root="checkpoints/NodeSTIDv2/PEMS_Combined_v2_${source_tag}_30_12_12"
  if [[ ! -d "$ckpt_root" ]]; then
    echo "[SKIP] Pretrain checkpoint root not found: ${ckpt_root}"
    continue
  fi

  best_ckpt=$(find "$ckpt_root" -type f -name "NodeSTIDv2_best_val_MAE.pt" | sort | head -n 1)
  if [[ -z "$best_ckpt" ]]; then
    echo "[SKIP] No best checkpoint found under: ${ckpt_root}"
    continue
  fi

  echo "[INFO] Using pretrained checkpoint: ${best_ckpt}"

  for target in "${datasets[@]}"; do
    if [[ "$target" == "$src_a" || "$target" == "$src_b" ]]; then
      continue
    fi

    gc_cfg="baselines/NodeSTID/${target}_GraphConditioned_From_${source_tag}_E30.py"
    if [[ ! -f "$gc_cfg" ]]; then
      echo "[SKIP] Missing graph config: ${gc_cfg}"
      continue
    fi

    echo "[RUN] Transfer ${source_tag} -> ${target}"
    NODESTID_PRETRAINED_PATH="$best_ckpt" run_train "$gc_cfg"
  done
done

echo "[DONE] Cross-dataset matrix finished."
"""


def write_if_missing(path: Path, content: str) -> bool:
    if path.exists():
        return False
    path.write_text(content, encoding="utf-8")
    return True


def main() -> None:
    created = []

    for a, b in PAIRS:
        tag = f"{a}_{b}"
        ds_list = f"['{a}', '{b}']"

        pre_file = BASE_DIR / f"PEMS_Combined_v2_{tag}_E30.py"
        if write_if_missing(pre_file, PRETRAIN_TEMPLATE.format(dataset_names=ds_list, tag=tag)):
            created.append(pre_file.name)

        for target in DATASETS:
            if target in {a, b}:
                continue

            gc_file = BASE_DIR / f"{target}_GraphConditioned_From_{tag}_E30.py"
            if write_if_missing(
                gc_file,
                GRAPH_TEMPLATE.format(dataset_names=ds_list, tag=tag, target=target, nodes=NODES[target]),
            ):
                created.append(gc_file.name)

    run_file = BASE_DIR / "run_cross_dataset_experiments.sh"
    if write_if_missing(run_file, RUN_SCRIPT):
        run_file.chmod(0o755)
        created.append(run_file.name)

    print("Created files:")
    for name in created:
        print(f"  - {name}")

    if not created:
        print("  (none; all files already existed)")


if __name__ == "__main__":
    main()
