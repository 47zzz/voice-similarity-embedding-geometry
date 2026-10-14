from configs.common import *

NAME = "fast_resnet34"
MODEL = "ResNetSE34L_feat"
NOUT = 512
N_MELS = 80
ENCODER_TYPE = "SAP"
MAX_EPOCH = 80
TEST_INTERVAL = 10
OPTIMIZER = "adam"
LR = 0.001
SCHEDULER = "steplr"
LR_DECAY = 0.95
WEIGHT_DECAY = 0
USE_SHARDS = True
