from configs.ecapa_tdnn import *

NAME = "ecapa_bottleneck"
LOSSES = ["aamsoftmax", "amsoftmax", "angleproto", "protofix"]
NOUTS = [512, 32, 10, 3]
FP32 = [(l, d) for l in ("aamsoftmax", "amsoftmax") for d in (10, 3)]
