import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from configs.common import VOX1_WAV, VOXSIM

CROP = 8 * 16000

MODELS = {
    "ecapa_sb":         "speechbrain/spkrec-ecapa-voxceleb (ECAPA-TDNN, SpeechBrain)",
    "wavlm_large_ft":   "yfyeung/wavlm-large-speaker-verification (WavLM-Large + ECAPA-TDNN, fine-tuned trunk)",
    "redimnet_b2_ftlm": "IDRnD/ReDimNet  b2 / vox2 / ft_lm",
    "redimnet_b6_ftlm": "IDRnD/ReDimNet  b6 / vox2 / ft_lm",
}


def build(name, device):
    if name == "ecapa_sb":
        from speechbrain.inference.speaker import EncoderClassifier
        m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
        return lambda w: m.encode_batch(w.to(device)).squeeze(1)
    if name.startswith("redimnet_"):
        _, size, _ = name.split("_")
        m = torch.hub.load("IDRnD/ReDimNet", "ReDimNet", model_name=size, train_type="ft_lm",
                           dataset="vox2", source="github").eval().to(device)
        return lambda w: m(w.to(device))
    if name == "wavlm_large_ft":
        sys.path.insert(0, str(Path(__file__).resolve().parent / "wavlm_large_sv"))
        from build import load
        m = load(device)
        return lambda w: m(w.to(device))
    raise ValueError(name)


def center_crop(y, n):
    if len(y) > n:
        o = (len(y) - n) // 2
        y = y[o:o + n]
    if len(y) < n:
        y = np.tile(y, int(np.ceil(n / max(len(y), 1))))[:n]
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    device = "cpu" if (a.cpu or not torch.cuda.is_available()) else "cuda:0"
    fwd = build(a.model, device)
    man = pd.read_csv(VOXSIM / "manifest_voxsim.csv")
    print(f"[{a.model}] {MODELS[a.model]}: {len(man)} clips on {device}", flush=True)
    embs, t0 = [], time.time()
    with torch.no_grad():
        for i, rel in enumerate(man.wav_path):
            y, sr = sf.read(str(VOX1_WAV / rel), dtype="float32")
            assert sr == 16000, (rel, sr)
            if y.ndim > 1:
                y = y.mean(1)
            w = torch.from_numpy(center_crop(y, CROP)).unsqueeze(0)
            embs.append(fwd(w).float().cpu().numpy().reshape(-1).astype(np.float32))
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(man)}  {el / 60:.1f} min", flush=True)
    E = np.stack(embs)
    out = Path(a.out) / "voxsim.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("emb", data=E, compression="gzip", compression_opts=4)
        sdt = h5py.string_dtype()
        f.create_dataset("utt_id", data=np.array(man.utt_id.astype(str).tolist(), dtype=object), dtype=sdt)
        f.create_dataset("speaker_id", data=np.array(man.speaker_id.astype(str).tolist(), dtype=object), dtype=sdt)
        f.create_dataset("wav_path", data=np.array([str(VOX1_WAV / p) for p in man.wav_path], dtype=object), dtype=sdt)
    print(f"wrote {out} {E.shape}", flush=True)


if __name__ == "__main__":
    main()
