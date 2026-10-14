from configs.common import *

NAME = "redimnet_b2"
MODEL = "ReDimNetB2_feat"
NOUT = 192
N_MELS = 72
ENCODER_TYPE = "ASTP"
MAX_EPOCH = 80
TEST_INTERVAL = 10
OPTIMIZER = "adamw"
LR = 0.0005
SCHEDULER = "steplr_ep"
LR_DECAY = 0.97
WEIGHT_DECAY = 0.01
USE_SHARDS = True
MIXEDPREC = False

LOSS_CFG = {k: dict(v) for k, v in LOSS_CFG.items()}
for _l in ("ge2e", "angleproto", "protofix", "supcon"):
    LOSS_CFG[_l]["batch_size"] = 200
