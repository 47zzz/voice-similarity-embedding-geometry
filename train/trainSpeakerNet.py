import argparse
import datetime
import logging
import os
import sys
import warnings
from pathlib import Path

import torch

from DatasetLoader import train_dataset_loader, train_dataset_sampler, worker_init_fn, feat_shard_dataset_loader
from SpeakerNet import ModelTrainer, SpeakerNet, WrappedModel
from tuneThreshold import tuneThresholdfromScore

warnings.simplefilter("ignore")
logging.basicConfig(
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    format='[%(levelname)s] :: %(asctime)s :: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
)


parser = argparse.ArgumentParser(description="SpeakerNet")

parser.add_argument('--max_frames',     type=int,   default=200,    help='Training crop length (frames)')
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Crop length of the in-training progress EER (frames)')
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size (number of speakers per batch for metric objectives)')
parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads')
parser.add_argument('--seed',           type=int,   default=10,     help='Random seed')

parser.add_argument('--test_interval',  type=int,   default=10,     help='Progress EER every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="",     help='Training objective: loss/<name>.py')

parser.add_argument('--optimizer',      type=str,   default="adam", help='optimizer/<name>.py')
parser.add_argument('--scheduler',      type=str,   default="steplr", help='scheduler/<name>.py')
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay factor')
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay')

parser.add_argument('--margin',         type=float, default=0.1,    help='Margin (AM-Softmax, AAM-Softmax)')
parser.add_argument('--scale',          type=float, default=30,     help='Scale (NSL, AM-Softmax, AAM-Softmax)')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Utterances per speaker per batch (metric objectives)')
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the classification head')

parser.add_argument('--initial_model',  type=str,   default="",     help='Initial weights (weights only)')
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Output directory (model/ and result/)')

parser.add_argument('--train_list',     type=str,   default="data/train_list.txt",  help='Training list: "<speaker> <relative path>" per line')
parser.add_argument('--test_list',      type=str,   default="data/veri_test.txt",   help='Trial list of the progress EER')
parser.add_argument('--train_path',     type=str,   default="data/voxceleb2/dev/wav", help='Root of the training wav files (raw-waveform training)')
parser.add_argument('--feat_shard_path', type=str,  default="",     help='If set, train on the precomputed per-speaker log-mel shards in this directory')
parser.add_argument('--test_path',      type=str,   default="data/voxceleb1/test/wav", help='Root of the test wav files')

parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      dest='log_input', action='store_true', help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='models/<name>.py')
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Pooling type (where the model has a choice)')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size')

parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')
parser.add_argument('--gpu_id',         type=int,   default=0,      help='GPU index')

args = parser.parse_args()


def _prune_checkpoints(model_dir, keep=1):
    d = Path(model_dir)
    models = sorted(d.glob("model0*.model"))
    for old in (models[:-keep] if keep > 0 else []):
        try:
            old.unlink()
        except OSError:
            pass
    for tmp in d.glob("model0*.model.tmp.*"):
        try:
            tmp.unlink()
        except OSError:
            pass


def main():
    logger = logging.getLogger('SpeakerNet')

    args.model_save_path  = str(Path(args.save_path) / "model")
    args.result_save_path = str(Path(args.save_path) / "result")
    Path(args.model_save_path).mkdir(parents=True, exist_ok=True)
    Path(args.result_save_path).mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu = 0

    file_handler = logging.FileHandler(Path(args.result_save_path) / "scores.txt", mode="a+")
    file_handler.setFormatter(logging.Formatter('[%(levelname)s] :: %(asctime)s :: %(message)s', datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    logger.info(f'Python Version: {sys.version}')
    logger.info(f'PyTorch Version: {torch.__version__}')
    logger.info(f'Save path: {args.save_path}')

    import random as _random
    import numpy as _np
    _random.seed(args.seed)
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    s = WrappedModel(SpeakerNet(**vars(args))).cuda(args.gpu)

    train_dataset = feat_shard_dataset_loader(**vars(args)) if args.feat_shard_path else train_dataset_loader(**vars(args))
    train_sampler = train_dataset_sampler(train_dataset, **vars(args))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    trainer = ModelTrainer(s, **vars(args))

    it = 1
    modelfiles = sorted(Path(args.model_save_path).glob('model0*.model'))
    loaded = None
    for cand in reversed(modelfiles):
        try:
            epoch = trainer.loadParameters(str(cand))
            loaded = cand
            break
        except Exception as e:
            logger.warning(f"Checkpoint {cand} failed to load ({e}); trying an earlier one.")
            try:
                cand.unlink()
            except OSError:
                pass
    if loaded is not None:
        logger.info(f"Model {loaded} resumed from previous state!")
        it = (epoch + 1) if epoch is not None else int(loaded.stem[5:]) + 1
    elif args.initial_model != "":
        trainer.loadParameters(args.initial_model, restore_optim=False)
        logger.info(f"Initialised from {args.initial_model} (weights only); training from epoch 1.")

    strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    with open(Path(args.result_save_path) / f'run{strtime}.cmd', 'w') as f:
        f.write(f'{args}')

    for it in range(it, args.max_epoch + 1):

        train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        loss, traineer = trainer.train_network(train_loader, verbose=True)

        logger.info(f"Epoch {it:d}, TEER/TAcc {traineer:2.2f}, TLOSS {loss:f}, LR {max(clr):f}")

        trainer.saveParameters(str(Path(args.model_save_path) / f"model{it:09d}.model"), epoch=it)
        _prune_checkpoints(args.model_save_path, keep=1)

        if it % args.test_interval == 0:
            sc, lab, _ = trainer.evaluateFromList(**vars(args))
            result = tuneThresholdfromScore(sc, lab, [1, 0.1])
            logger.info(f"Epoch {it:d}, VEER {result[1]:2.4f}")
            with open(Path(args.model_save_path) / f"model{it:09d}.eer", 'w') as eerfile:
                eerfile.write(f'{result[1]:2.4f}')


if __name__ == '__main__':
    main()
