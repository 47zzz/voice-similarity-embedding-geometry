import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

EXPECT_SPK = 5994
EXPECT_UTT = 1_092_009

_FB = None
def _mel_init(n_mels):
    global _FB
    import torch, torchaudio
    _FB = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
        window_fn=torch.hamming_window, n_mels=n_mels)

def _decode_f32(path):
    out = subprocess.run(["ffmpeg", "-v", "quiet", "-i", str(path), "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32)

def _mel_speaker(job):
    import torch
    spk_dir, out_dir = Path(job[0]), Path(job[1])
    spk = spk_dir.name
    onpy, oidx = out_dir / f"{spk}.npy", out_dir / f"{spk}.idx.json"
    if onpy.exists() and oidx.exists():
        return (1, 0)
    aac_root = spk_dir.parent
    mels, index, cur, n = [], {}, 0, 0
    for m4a in sorted(spk_dir.rglob("*.m4a")):
        rel = m4a.relative_to(aac_root).with_suffix("").as_posix()
        wav = _decode_f32(m4a)
        if wav.size < 400:
            continue
        x = torch.from_numpy(wav.copy()).unsqueeze(0)
        with torch.no_grad():
            mel = (_FB(x) + 1e-6).log().squeeze(0).numpy().astype(np.float16)
        index[rel] = [cur, cur + mel.shape[1]]
        mels.append(mel); cur += mel.shape[1]; n += 1
    if not mels:
        return (0, 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(onpy, np.concatenate(mels, axis=1).astype(np.float16))
    oidx.write_text(json.dumps(index), encoding="utf-8")
    return (1, n)

_FB80 = _FB72 = None
def _mel_init_dual():
    global _FB80, _FB72
    import torch, torchaudio
    def _mk(nm):
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
            window_fn=torch.hamming_window, n_mels=nm)
    _FB80, _FB72 = _mk(80), _mk(72)

def _mel_speaker_dual(job):
    import torch
    spk_dir, out80, out72 = Path(job[0]), Path(job[1]), Path(job[2])
    spk = spk_dir.name
    done80 = (out80 / f"{spk}.npy").exists() and (out80 / f"{spk}.idx.json").exists()
    done72 = (out72 / f"{spk}.npy").exists() and (out72 / f"{spk}.idx.json").exists()
    if done80 and done72:
        return (1, 0)
    aac_root = spk_dir.parent
    mels80, mels72, index, cur, n = [], [], {}, 0, 0
    for m4a in sorted(spk_dir.rglob("*.m4a")):
        rel = m4a.relative_to(aac_root).with_suffix("").as_posix()
        wav = _decode_f32(m4a)
        if wav.size < 400:
            continue
        x = torch.from_numpy(wav.copy()).unsqueeze(0)
        with torch.no_grad():
            m80 = (_FB80(x) + 1e-6).log().squeeze(0).numpy().astype(np.float16)
            m72 = (_FB72(x) + 1e-6).log().squeeze(0).numpy().astype(np.float16)
        index[rel] = [cur, cur + m80.shape[1]]
        mels80.append(m80); mels72.append(m72); cur += m80.shape[1]; n += 1
    if not mels80:
        return (0, 0)
    out80.mkdir(parents=True, exist_ok=True); out72.mkdir(parents=True, exist_ok=True)
    np.save(out80 / f"{spk}.npy", np.concatenate(mels80, axis=1).astype(np.float16))
    (out80 / f"{spk}.idx.json").write_text(json.dumps(index), encoding="utf-8")
    np.save(out72 / f"{spk}.npy", np.concatenate(mels72, axis=1).astype(np.float16))
    (out72 / f"{spk}.idx.json").write_text(json.dumps(index), encoding="utf-8")
    return (1, n)

def _wav_one(job):
    src, dst = job
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
                         str(dst)], capture_output=True).returncode
    return 1 if (rc == 0 and dst.exists()) else 0


def _run(fn, jobs, workers, tag, init=None, initargs=()):
    print(f"[{tag}] {len(jobs)} jobs, {workers} workers", flush=True)
    t0, ok, extra = time.time(), 0, 0
    with Pool(workers, initializer=init, initargs=initargs) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, jobs, chunksize=4), 1):
            if isinstance(r, tuple): ok += r[0]; extra += r[1]
            else: ok += r
            if i % 500 == 0:
                el = time.time() - t0
                print(f"[{tag}] {i}/{len(jobs)} ok={ok} {el/60:.1f}m ETA={el/i*(len(jobs)-i)/60:.1f}m", flush=True)
    print(f"[{tag}] done: {ok}/{len(jobs)} ({extra} utts) in {(time.time()-t0)/60:.1f}m", flush=True)
    return ok, extra


def step_unzip(aac_dir, zip_path, extract):
    aac_dir, zip_path, extract = Path(aac_dir), Path(zip_path), Path(extract)
    parts = sorted(aac_dir.glob("vox2_dev_aac_part*"))
    if not parts:
        sys.exit(f"[unzip] no vox2_dev_aac_part* under {aac_dir}")
    total = sum(p.stat().st_size for p in parts)
    if not (zip_path.exists() and zip_path.stat().st_size == total):
        print(f"[unzip] concat {len(parts)} parts -> {zip_path} ({total/1e9:.1f} GB)", flush=True)
        with open(zip_path, "wb") as out:
            for p in parts:
                print(f"   + {p.name}", flush=True)
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out, length=32 * 1024 * 1024)
    extract.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] extracting {zip_path} -> {extract}", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract)
    aac_root = extract / "dev" / "aac"
    print(f"[unzip] AAC root = {aac_root}  ({sum(1 for _ in aac_root.rglob('*.m4a'))} .m4a)", flush=True)


def step_mel(aac_root, out_dir, n_mels, workers):
    spks = [d for d in sorted(Path(aac_root).iterdir()) if d.is_dir()]
    jobs = [(str(d), str(out_dir)) for d in spks]
    _run(_mel_speaker, jobs, workers, f"mel{n_mels}", init=_mel_init, initargs=(n_mels,))


def step_meldual(aac_root, out80, out72, workers):
    spks = [d for d in sorted(Path(aac_root).iterdir()) if d.is_dir()]
    jobs = [(str(d), str(out80), str(out72)) for d in spks]
    _run(_mel_speaker_dual, jobs, workers, "mel80+72", init=_mel_init_dual)


def step_wav(aac_root, out_dir, workers):
    aac_root, out_dir = Path(aac_root), Path(out_dir)
    jobs = [(str(m), str(out_dir / m.relative_to(aac_root).with_suffix(".wav")))
            for m in aac_root.rglob("*.m4a")]
    _run(_wav_one, jobs, workers, "wav")


def step_verify(out_dir, kind):
    out_dir = Path(out_dir)
    if kind == "shards":
        n = len(list(out_dir.glob("*.npy")))
        print(f"[verify] {out_dir.name}: {n} speaker shards (expect {EXPECT_SPK})"
              + ("  OK" if n == EXPECT_SPK else "  ⚠️ MISMATCH"), flush=True)
    else:
        n = sum(1 for _ in out_dir.rglob("*.wav"))
        print(f"[verify] {out_dir.name}: {n} wavs (full set expects {EXPECT_UTT})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("unzip"); p.add_argument("--aac_dir", required=True); p.add_argument("--zip", required=True); p.add_argument("--extract", required=True)
    p = sub.add_parser("mel");   p.add_argument("--aac", required=True); p.add_argument("--n_mels", type=int, required=True); p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=18)
    p = sub.add_parser("meldual"); p.add_argument("--aac", required=True); p.add_argument("--out80", required=True); p.add_argument("--out72", required=True); p.add_argument("--workers", type=int, default=18)
    p = sub.add_parser("wav");   p.add_argument("--aac", required=True); p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=18)
    p = sub.add_parser("verify"); p.add_argument("--out", required=True); p.add_argument("--kind", choices=["shards", "wav"], required=True)
    a = ap.parse_args()
    if a.step == "unzip":
        step_unzip(a.aac_dir, a.zip, a.extract)
    elif a.step == "mel":
        step_mel(a.aac, a.out, a.n_mels, a.workers); step_verify(a.out, "shards")
    elif a.step == "meldual":
        step_meldual(a.aac, a.out80, a.out72, a.workers)
        step_verify(a.out80, "shards"); step_verify(a.out72, "shards")
    elif a.step == "wav":
        step_wav(a.aac, a.out, a.workers); step_verify(a.out, "wav")
    elif a.step == "verify":
        step_verify(a.out, a.kind)


if __name__ == "__main__":
    main()
