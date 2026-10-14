import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.metrics import alignment, eer_percent, eer_percent_pairs, effective_dim, load_h5, pair_cosines, trial_scores


def score(voxsim_h5, vox1o_h5, voxsim_dir, trials):
    row = {}
    if vox1o_h5 is not None:
        E, utt, _ = load_h5(vox1o_h5)
        sc, lab = trial_scores(E, utt, Path(trials).read_text(encoding="utf-8").splitlines())
        row["eer_vox1o"] = eer_percent(sc, lab)
        row["n_trials"] = int(len(sc))
    if voxsim_h5 is not None:
        E, utt, spk = load_h5(voxsim_h5)
        pairs = pd.read_csv(Path(voxsim_dir) / "pairs_mean.csv")
        s = pair_cosines(E, utt, pairs.clip1.values, pairs.clip2.values)
        row["rho_align"] = alignment(s, pairs, "diffspk")
        row["d_eff"] = effective_dim(E[np.isfinite(E).all(1)], np.asarray(spk)[np.isfinite(E).all(1)])
        ok = np.isfinite(s)
        row["eer_voxsim"] = eer_percent_pairs(s[ok], pairs["same"].values.astype(int)[ok])
        row["n_pairs"] = int(ok.sum())
        row["dim"] = int(E.shape[1])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", help="directory holding voxsim.h5 and vox1o.h5 (from eval/extract_embeddings.py)")
    ap.add_argument("--voxsim_h5")
    ap.add_argument("--vox1o_h5")
    ap.add_argument("--voxsim_dir", default=str(ROOT / "data" / "voxsim"))
    ap.add_argument("--trials", default=str(ROOT / "data" / "lists" / "veri_test2.txt"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--append", default=None, help="append the row to this CSV")
    a = ap.parse_args()
    vs, vo = a.voxsim_h5, a.vox1o_h5
    if a.emb_dir:
        d = Path(a.emb_dir)
        vs = vs or (str(d / "voxsim.h5") if (d / "voxsim.h5").exists() else None)
        vo = vo or (str(d / "vox1o.h5") if (d / "vox1o.h5").exists() else None)
    if vs is None and vo is None:
        sys.exit("nothing to score: give --emb_dir or --voxsim_h5 / --vox1o_h5")
    row = {"name": a.name or (a.emb_dir or vs or vo)}
    row.update(score(vs, vo, a.voxsim_dir, a.trials))
    print(json.dumps(row, indent=2))
    if a.append:
        p = Path(a.append)
        pd.DataFrame([row]).to_csv(p, mode="a", header=not p.exists(), index=False)


if __name__ == "__main__":
    main()
