import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINER = ROOT / "train"

LOSSES = ["protofix", "angleproto", "normface", "aamsoftmax", "amsoftmax", "ge2e", "supcon"]
LOSS_NAME = {
    "protofix":   "Prototypical",
    "angleproto": "Angular Prototypical",
    "normface":   "NSL",
    "aamsoftmax": "AAM-Softmax",
    "amsoftmax":  "AM-Softmax",
    "ge2e":       "GE2E",
    "supcon":     "SupCon",
}
SEEDS = [10, 11, 12]
N_DEV_SPEAKERS = 5994

LOSS_CFG = {
    "normface":   dict(nClasses=N_DEV_SPEAKERS, nPerSpeaker=1, scale=30, batch_size=200),
    "amsoftmax":  dict(nClasses=N_DEV_SPEAKERS, nPerSpeaker=1, margin=0.2, scale=30, batch_size=200),
    "aamsoftmax": dict(nClasses=N_DEV_SPEAKERS, nPerSpeaker=1, margin=0.2, scale=30, batch_size=200),
    "ge2e":       dict(nPerSpeaker=2, batch_size=400),
    "angleproto": dict(nPerSpeaker=2, batch_size=400),
    "protofix":   dict(nPerSpeaker=2, batch_size=400),
    "supcon":     dict(nPerSpeaker=2, batch_size=400),
}

MAX_FRAMES = 200
MAX_SEG_PER_SPK = 100000
EVAL_FRAMES = 400
MIXEDPREC = True
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "6"))

DATA = Path(os.environ.get("DATA_ROOT", str(ROOT / "data")))
VOX1_WAV = Path(os.environ.get("VOX1_WAV", str(DATA / "vox1_wav")))
VOX2_WAV = Path(os.environ.get("VOX2_WAV", str(DATA / "vox2_wav")))
LISTS = DATA / "lists"
TRAIN_LIST_NPY = LISTS / "vox2_dev_train.txt"
TRAIN_LIST_WAV = LISTS / "vox2_dev_train_wav.txt"
TEST_LIST = LISTS / "veri_test2.txt"
VOXSIM = Path(os.environ.get("VOXSIM_DIR", str(DATA / "voxsim")))


def feat_shards(n_mels):
    return Path(os.environ.get("FEAT_SHARDS", str(DATA / f"feat_shards_{n_mels}")))
