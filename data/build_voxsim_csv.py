import argparse
from pathlib import Path

import pandas as pd

COLS = ["clip1", "clip2", "same", "listener", "rating"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxsim_trainer", required=True, help="path of the cloned kaistmm/voxsim_trainer repository")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "voxsim"))
    a = ap.parse_args()
    src = Path(a.voxsim_trainer) / "data"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(src / "voxsim_train_list_raw.txt", header=None, names=COLS)
    mean = pd.read_csv(src / "voxsim_train_list_mean.txt", header=None, names=COLS)
    test = pd.read_csv(src / "voxsim_test_list.txt", header=None, names=COLS)

    pm = pd.concat([mean, test], ignore_index=True)
    pm = pm[["clip1", "clip2", "same", "rating"]]
    pm["same"] = pm.same.astype(int)
    pm.to_csv(out / "pairs_mean.csv", index=False)

    pr = raw[["clip1", "clip2", "same", "rating", "listener"]].copy()
    pr["same"] = pr.same.astype(int)
    pr["rating"] = pr.rating.astype(float)
    pr.to_csv(out / "pairs_raw.csv", index=False)

    clips = sorted(set(pm.clip1) | set(pm.clip2))
    man = pd.DataFrame({"utt_id": clips, "speaker_id": [c.split("/")[0] for c in clips], "wav_path": clips})
    man.to_csv(out / "manifest_voxsim.csv", index=False)

    print(f"pairs_mean.csv: {len(pm)} pairs ({int((pm.same == 0).sum())} different-speaker, {int((pm.same == 1).sum())} same-speaker)")
    print(f"pairs_raw.csv: {len(pr)} ratings from {pr.listener.nunique()} listener ids")
    print(f"manifest_voxsim.csv: {len(man)} clips, {man.speaker_id.nunique()} speakers -> {out}")


if __name__ == "__main__":
    main()
