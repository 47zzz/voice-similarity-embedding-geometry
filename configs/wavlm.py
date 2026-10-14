from configs.common import *

NAME = "wavlm"
MODEL = "WavLM_ECAPA_feat"
NOUT = 192
N_MELS = 80
ENCODER_TYPE = "ASP"
TEST_INTERVAL = 5
OPTIMIZER = "adam"
SCHEDULER = "steplr_ep"
LR_DECAY = 0.97
WEIGHT_DECAY = 2e-5
USE_SHARDS = False

PHASES = {
    "phase1": dict(epochs=20, lr=0.001,  finetune="0"),
    "phase2": dict(epochs=5,  lr=0.0001, finetune="1"),
}
MAX_EPOCH, LR = PHASES["phase1"]["epochs"], PHASES["phase1"]["lr"]
