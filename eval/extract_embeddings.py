import argparse
import importlib
import itertools
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

MIN_SECONDS = 2.0


def load_wav(path, sr=16000):
    wav, file_sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if file_sr != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    if len(wav) < MIN_SECONDS * sr:
        wav = np.pad(wav, (0, int(MIN_SECONDS * sr) - len(wav)))
    return np.ascontiguousarray(wav, dtype=np.float32)


def build_model(C, nout, ckpt, device):
    MainModel = importlib.import_module(f"models.{C.MODEL}").MainModel
    m = MainModel(nOut=nout, encoder_type=C.ENCODER_TYPE, n_mels=C.N_MELS, log_input=True)
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    backbone = {k[len("__S__."):]: v for k, v in sd.items() if k.startswith("__S__.")}
    try:
        res = m.load_state_dict(backbone, strict=False)
    except RuntimeError as e:
        sys.exit(f"[abort] checkpoint does not match {C.MODEL} at nOut={nout} (pass --nout): {e}")
    if res.missing_keys:
        sys.exit(f"[abort] checkpoint does not match {C.MODEL} at nOut={nout}: missing {res.missing_keys[:5]}")
    return m.to(device).eval()


def voxsim_manifest(C):
    man = pd.read_csv(C.VOXSIM / "manifest_voxsim.csv")
    return pd.DataFrame({"utt_id": man.utt_id, "speaker_id": man.speaker_id,
                         "wav_path": [str(C.VOX1_WAV / p) for p in man.wav_path]})


def vox1o_manifest(C):
    lines = Path(C.TEST_LIST).read_text(encoding="utf-8").splitlines()
    files = sorted(set(itertools.chain(*[ln.split()[-2:] for ln in lines if ln.strip()])))
    return pd.DataFrame({"utt_id": files, "speaker_id": [f.split("/")[0] for f in files],
                         "wav_path": [str(C.VOX1_WAV / f) for f in files]})


def embed_all(model, device, manifest, out_path, dim, name):
    N = len(manifest)
    full = np.full((N, dim), np.nan, dtype=np.float32)
    fails = []
    t0 = time.time()
    with torch.no_grad():
        for i, row in enumerate(tqdm(manifest.itertuples(index=False), total=N, desc=name)):
            try:
                x = torch.from_numpy(load_wav(row.wav_path)).unsqueeze(0).to(device)
                full[i] = model(x).squeeze(0).float().cpu().numpy()
            except Exception as ex:
                fails.append(f"{row.utt_id}\t{ex}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("emb", data=full, compression="gzip", compression_opts=4)
        sdt = h5py.string_dtype()
        for col in ("utt_id", "speaker_id", "wav_path"):
            f.create_dataset(col, data=np.array(manifest[col].astype(str).tolist(), dtype=object), dtype=sdt)
    if fails:
        out_path.with_suffix(".fails.txt").write_text("\n".join(fails), encoding="utf-8")
    print(f"[{name}] wrote {out_path} ({N} clips, {len(fails)} failed, {(time.time() - t0) / 60:.1f} min)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    choices=["ecapa_tdnn", "fast_resnet34", "redimnet_b2", "wavlm", "ecapa_bottleneck"])
    ap.add_argument("--ckpt", required=True, help="final-epoch checkpoint (model0000000NN.model)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--nout", type=int, default=None, help="embedding width if not the config default")
    ap.add_argument("--datasets", nargs="*", default=["voxsim", "vox1o"], choices=["voxsim", "vox1o"])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="embed only the first N clips of each set (quick check of a checkpoint)")
    a = ap.parse_args()

    C = importlib.import_module(f"configs.{a.config}")
    nout = a.nout or C.NOUT
    device = "cpu" if (a.cpu or not torch.cuda.is_available()) else "cuda:0"
    model = build_model(C, nout, a.ckpt, device)
    out = Path(a.out)
    if "voxsim" in a.datasets:
        embed_all(model, device, voxsim_manifest(C).head(a.limit), out / "voxsim.h5", nout, "voxsim")
    if "vox1o" in a.datasets:
        embed_all(model, device, vox1o_manifest(C).head(a.limit), out / "vox1o.h5", nout, "vox1o")


if __name__ == "__main__":
    main()
