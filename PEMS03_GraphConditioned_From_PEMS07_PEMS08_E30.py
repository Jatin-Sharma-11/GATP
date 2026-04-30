import glob
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
DATA_NAME = 'PEMS03'
SOURCE_DATASET_NAMES = ['PEMS07', 'PEMS08']
SOURCE_TAG = 'PEMS07_PEMS08'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

NUM_NODES = 358
adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "doubletransition")

############################## Pretrained Backbone ##############################
def _resolve_default_pretrained_path() -> str:
    ckpt_root = os.path.join(
        'checkpoints',
        'NodeSTIDv2',
        f'PEMS_Combined_v2_{SOURCE_TAG}_30_12_12',
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
MODEL_PARAM = {
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
}
NUM_EPOCHS = 40

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = f'NodeSTID-GC: transfer from {SOURCE_TAG} to {DATA_NAME} (100 epochs)'
CFG.GPU_NUM = 1
CFG.RUNNER = GraphConditionedRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

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
CFG.METRICS.FUNCS = EasyDict({
    'MAE': masked_mae,
    'MAPE': masked_mape,
    'RMSE': masked_rmse,
})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([f'{DATA_NAME}_from_{SOURCE_TAG}', str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0.0001,
}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 50, 80],
    "gamma": 0.5
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}

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
