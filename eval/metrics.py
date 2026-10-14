import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve


def l2(X):
    X = np.asarray(X, dtype=np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def load_h5(path):
    import h5py
    with h5py.File(path, "r") as f:
        E = f["emb"][:].astype(np.float64)
        utt = [u.decode() if isinstance(u, bytes) else str(u) for u in f["utt_id"][:]]
        spk = [s.decode() if isinstance(s, bytes) else str(s) for s in f["speaker_id"][:]]
    return E, utt, spk


def eer_percent(scores, labels):
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float(max(fpr[i], fnr[i]) * 100)


def eer_percent_pairs(scores, labels):
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    order = np.argsort(-scores)
    la = labels[order]
    n_tgt = la.sum()
    n_non = len(la) - n_tgt
    fnr = (n_tgt - np.cumsum(la)) / n_tgt
    fpr = np.cumsum(1 - la) / n_non
    i = np.argmin(np.abs(fnr - fpr))
    return float((fnr[i] + fpr[i]) / 2 * 100)


def pair_cosines(E, utt_ids, clip1, clip2):
    idx = {u: i for i, u in enumerate(utt_ids)}
    i1 = np.array([idx.get(a, -1) for a in clip1])
    i2 = np.array([idx.get(b, -1) for b in clip2])
    ok = (i1 >= 0) & (i2 >= 0)
    En = l2(E)
    s = np.full(len(i1), np.nan)
    s[ok] = np.einsum("ij,ij->i", En[i1[ok]], En[i2[ok]])
    return s


def trial_scores(E, utt_ids, trial_lines):
    idx = {u: i for i, u in enumerate(utt_ids)}
    En = l2(E)
    sc, lab = [], []
    for ln in trial_lines:
        t = ln.split()
        if len(t) < 3 or t[1] not in idx or t[2] not in idx:
            continue
        s = float(En[idx[t[1]]] @ En[idx[t[2]]])
        if np.isfinite(s):
            sc.append(s)
            lab.append(int(t[0]))
    return np.array(sc), np.array(lab)


def alignment(sims, pairs, scope="diffspk"):
    m = np.isfinite(sims)
    if scope == "diffspk":
        m &= pairs["same"].values.astype(int) == 0
    return float(spearmanr(sims[m], pairs["rating"].values[m].astype(float)).correlation)


def effective_dim(E, speaker_ids):
    En = l2(E)
    spk = np.asarray(speaker_ids)
    C = np.vstack([En[spk == s].mean(0) for s in sorted(set(spk.tolist()))])
    C = l2(C)
    Xc = C - C.mean(0)
    ev = np.clip(np.linalg.eigvalsh(Xc.T @ Xc / len(Xc)), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())
