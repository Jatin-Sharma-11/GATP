import os
import sys

from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse

from .arch import NodeSTIDv2
from .dataset import CombinedNodeWiseDatasetV2
from .runner import NodeWiseTimeSeriesForecastingRunner
from .scaler import CombinedZScoreScaler

############################## Hot Parameters ##############################
# Evaluate pretrained combined model on PEMS08 only
DATASET_NAMES = ['PEMS04']       # single dataset for evaluation
DATASET_ID_INDEX = 1             # PEMS08 was index 0 during combined training

INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
NORM_EACH_CHANNEL = False
RESCALE = True
NULL_VAL = 0.0

# Must match the pretrained model architecture exactly
MODEL_ARCH = NodeSTIDv2
MODEL_PARAM = {
    "input_len": INPUT_LEN,
    "input_dim": 3,              # flow, time_of_day, day_of_week
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
    "num_datasets": 4,           # pretrained with 4 datasets
    "dataset_id_dim": 32,
}
NUM_EPOCHS = 0                   # no training, evaluation only

# Pretrained checkpoint path
CKPT_PATH = os.path.join(
    'checkpoints',
    'NodeSTIDv2',
    'PEMS_Combined_v2_100_12_12',
    'ed4bbd9b3ecee86e4f415b9c584b6b99',
    'NodeSTIDv2_best_val_MAE.pt'
)

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'Evaluate pretrained NodeSTIDv2 (combined) on PEMS08 test set'
CFG.GPU_NUM = 1

# Runner
CFG.RUNNER = NodeWiseTimeSeriesForecastingRunner

############################## Dataset Configuration ##############################
# Use CombinedNodeWiseDatasetV2 with only PEMS08 so that dataset_id=0 is
# automatically appended as the 4th channel, matching the pretrained model's
# expected input format.
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = 'PEMS08_Eval'
CFG.DATASET.TYPE = CombinedNodeWiseDatasetV2
CFG.DATASET.PARAM = EasyDict({
    'dataset_names': DATASET_NAMES,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
})

############################## Scaler Configuration ##############################
# Use CombinedZScoreScaler with all 4 datasets so mean/std match training.
# This ensures the pretrained model sees data scaled exactly as during training.
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = CombinedZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_names': ['PEMS08', 'PEMS03', 'PEMS04', 'PEMS07'],
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]   # flow, tod, dow, dataset_id
CFG.MODEL.TARGET_FEATURES = [0]              # predict flow only

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
    '_'.join(['PEMS08_Eval_Pretrained', str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae

# Optimizer (required by framework even for eval-only)
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

# Train data loader
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
