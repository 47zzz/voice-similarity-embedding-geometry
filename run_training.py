import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_dir(C, loss, seed, nout, phase):
    name = f"{loss}_s{seed}" if nout == C.NOUT else f"{loss}_d{nout}_s{seed}"
    d = ROOT / "checkpoints" / C.NAME
    return d / phase / name if phase else d / name


def latest_ckpt(save):
    models = sorted((save / "model").glob("model0*.model")) if (save / "model").exists() else []
    return models[-1] if models else None


def build_cmd(C, loss, seed, nout, phase, gpu, dry_run=False):
    epochs, lr = C.MAX_EPOCH, C.LR
    if phase:
        epochs, lr = C.PHASES[phase]["epochs"], C.PHASES[phase]["lr"]
    save = run_dir(C, loss, seed, nout, phase)
    cmd = [sys.executable, "trainSpeakerNet.py",
           "--model", C.MODEL, "--encoder_type", C.ENCODER_TYPE, "--log_input",
           "--n_mels", str(C.N_MELS), "--nOut", str(nout),
           "--max_frames", str(C.MAX_FRAMES), "--eval_frames", str(C.EVAL_FRAMES),
           "--max_seg_per_spk", str(C.MAX_SEG_PER_SPK),
           "--trainfunc", loss,
           "--optimizer", C.OPTIMIZER, "--scheduler", C.SCHEDULER,
           "--lr", str(lr), "--lr_decay", str(C.LR_DECAY), "--weight_decay", str(C.WEIGHT_DECAY),
           "--max_epoch", str(epochs), "--test_interval", str(C.TEST_INTERVAL),
           "--nDataLoaderThread", str(C.NUM_WORKERS),
           "--test_list", str(C.TEST_LIST), "--test_path", str(C.VOX1_WAV),
           "--save_path", str(save), "--seed", str(seed), "--gpu_id", str(gpu)]
    if C.USE_SHARDS:
        cmd += ["--train_list", str(C.TRAIN_LIST_NPY), "--feat_shard_path", str(C.feat_shards(C.N_MELS))]
    else:
        cmd += ["--train_list", str(C.TRAIN_LIST_WAV), "--train_path", str(C.VOX2_WAV)]
    for k, v in C.LOSS_CFG[loss].items():
        cmd += [f"--{k}", str(v)]
    if C.MIXEDPREC and (loss, nout) not in getattr(C, "FP32", ()):
        cmd += ["--mixedprec"]
    if phase == "phase2":
        p1 = latest_ckpt(run_dir(C, loss, seed, nout, "phase1"))
        if p1 is not None:
            cmd += ["--initial_model", str(p1)]
        elif latest_ckpt(save) is not None:
            pass
        elif dry_run:
            cmd += ["--initial_model", "<checkpoints/wavlm/phase1/.../model0000000020.model>"]
        else:
            sys.exit(f"[abort] phase2 of {loss} seed {seed} needs the phase-1 checkpoint; train phase1 first")
    return cmd, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    choices=["ecapa_tdnn", "fast_resnet34", "redimnet_b2", "wavlm", "ecapa_bottleneck"])
    ap.add_argument("--losses", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--nout", type=int, default=None, help="embedding width (ecapa_bottleneck: 512 / 32 / 10 / 3)")
    ap.add_argument("--phase", choices=["phase1", "phase2"], default=None, help="wavlm only")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    C = importlib.import_module(f"configs.{a.config}")
    if a.config == "wavlm" and a.phase is None:
        sys.exit("[abort] --config wavlm needs --phase phase1 or phase2")
    if a.config != "wavlm" and a.phase is not None:
        sys.exit("[abort] --phase applies to --config wavlm only")
    nout = a.nout or C.NOUT
    if a.config == "ecapa_bottleneck" and nout not in C.NOUTS:
        sys.exit(f"[abort] ecapa_bottleneck expects --nout in {C.NOUTS}")
    losses = a.losses or C.LOSSES
    bad = [l for l in losses if l not in C.LOSSES]
    if bad:
        sys.exit(f"[abort] unknown objective(s) {bad}; choose from {C.LOSSES}")
    seeds = a.seeds or C.SEEDS

    env = dict(os.environ)
    if a.config == "wavlm":
        env.update(WAVLM_FINETUNE=C.PHASES[a.phase]["finetune"])

    for loss in losses:
        for seed in seeds:
            save = run_dir(C, loss, seed, nout, a.phase)
            if (save / "DONE").exists():
                print(f"[skip] {save.relative_to(ROOT)} already DONE")
                continue
            cmd, save = build_cmd(C, loss, seed, nout, a.phase, a.gpu, a.dry_run)
            print(f"[run ] {save.relative_to(ROOT)}\n       " + " ".join(cmd), flush=True)
            if a.dry_run:
                continue
            save.mkdir(parents=True, exist_ok=True)
            with open(save / "train.log", "a", encoding="utf-8") as lf:
                rc = subprocess.call(cmd, cwd=str(C.TRAINER), stdout=lf, stderr=subprocess.STDOUT, env=env)
            if rc == 0:
                (save / "DONE").write_text("finished\n")
            print(f"[{'done' if rc == 0 else 'FAILED rc=' + str(rc)}] {save.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
