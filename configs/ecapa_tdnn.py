from configs.common import *

NAME = "ecapa_tdnn"
MODEL = "ECAPA_TDNN_feat"
NOUT = 192
N_MELS = 80
ENCODER_TYPE = "SAP"
MAX_EPOCH = 80
TEST_INTERVAL = 10
OPTIMIZER = "adam"
LR = 0.001
SCHEDULER = "steplr_ep"
LR_DECAY = 0.97
WEIGHT_DECAY = 2e-5
USE_SHARDS = True
