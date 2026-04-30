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
DATA_NAME = 'PEMS04'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']           # 12
OUTPUT_LEN = regular_settings['OUTPUT_LEN']          # 12
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

NUM_NODES = 307

# Load adjacency matrix for graph conditioning
adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "doubletransition")

############################## Pretrained Backbone ##############################
# Path to the pretrained NodeSTIDv2 checkpoint from combined pretraining.
# Update this path to point to your best pretrained checkpoint.

PRETRAINED_PATH = os.path.join(
    'checkpoints', 'NodeSTIDv2',
    'PEMS_Combined_v2_100_12_12',
    'ed4bbd9b3ecee86e4f415b9c584b6b99',
    'NodeSTIDv2_best_val_MAE.pt'
)
FREEZE_BACKBONE = True  # Set False to finetune the whole model end-to-end

############################## Model Configuration ##############################
MODEL_ARCH = NodeSTIDGraphConditioned
MODEL_PARAM = {
    # ---- Backbone params (must match pretrained NodeSTIDv2) ----
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
    "num_datasets": 4,
    "dataset_id_dim": 32,
    "num_nodes": NUM_NODES,
    "freeze_backbone": FREEZE_BACKBONE,
    "default_dataset_id": 0,   # PEMS08 was index 0 in pretraining ['PEMS08','PEMS03','PEMS04','PEMS07']
    # ---- Graph conditioning params ----
    "supports": [torch.tensor(i) for i in adj_mx],
    "use_adaptive_adj": True,
    "node_emb_dim": 10,
    "num_graph_layers": 3,
    "gcn_order": 2,
    "gcn_dropout": 0.1,
    # graph_hidden_dim defaults to backbone_hidden_dim if not set
}
NUM_EPOCHS = 100

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'NodeSTID-GC: Pretrained backbone + Graph Conditioning on PEMS04'
CFG.GPU_NUM = 1

# Runner — loads pretrained backbone and handles standard [B, L, N, C] data
CFG.RUNNER = GraphConditionedRunner

############################## Dataset Configuration ##############################
# Standard dataset — full graph [B, L, N, C], NOT node-wise
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
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]   # flow, tod, dow (dataset_id not needed per-dataset)
CFG.MODEL.TARGET_FEATURES = [0]           # predict flow only
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
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae

# Optimizer
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0.0001,
}

# Learning rate scheduler
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 50, 80],
    "gamma": 0.5
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}

# Data loader — standard graph batch sizes
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
