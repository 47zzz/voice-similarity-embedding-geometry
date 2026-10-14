import importlib.util
import os
import sys
import types
import urllib.request
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download

UNISPEECH_COMMIT = "6112826ac13a4327f4c9a7afa2a505e35b763514"
UNISPEECH_URL = (f"https://raw.githubusercontent.com/microsoft/UniSpeech/{UNISPEECH_COMMIT}/"
                 "downstreams/speaker_verification/models/ecapa_tdnn.py")
CACHE = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "voice-similarity-embedding-geometry"

torchaudio.set_audio_backend = lambda *a, **k: None
if not hasattr(torchaudio, "sox_effects"):
    _m = types.ModuleType("torchaudio.sox_effects")
    _m.apply_effects_tensor = _m.apply_effects_file = None
    torchaudio.sox_effects = _m
    sys.modules["torchaudio.sox_effects"] = _m


def unispeech_ecapa_tdnn():
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / f"unispeech_ecapa_tdnn_{UNISPEECH_COMMIT[:8]}.py"
    if not target.exists():
        src = urllib.request.urlopen(UNISPEECH_URL, timeout=60).read().decode("utf-8")
        src = src.replace("from .utils import UpstreamExpert",
                          "UpstreamExpert = None  # package-relative import removed; only the torch.hub path is used")
        target.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("unispeech_ecapa_tdnn", target)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ECAPA_TDNN


def load(device="cpu"):
    import s3prl.hub as hub
    ECAPA_TDNN = unispeech_ecapa_tdnn()
    ckpt = hf_hub_download("yfyeung/wavlm-large-speaker-verification", "wavlm-large.pt")
    _real = torch.hub.load
    torch.hub.load = lambda *a, **k: hub.wavlm_large()
    try:
        m = ECAPA_TDNN(feat_dim=1024, emb_dim=256, feat_type="wavlm_large",
                       feature_selection="hidden_states", update_extract=False)
    finally:
        torch.hub.load = _real
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    r = m.load_state_dict(sd, strict=False)
    stray = [k for k in r.unexpected_keys if not k.startswith("loss_calculator")]
    assert not stray, stray[:5]
    assert not r.missing_keys, r.missing_keys[:5]
    return m.eval().to(device)


if __name__ == "__main__":
    m = load()
    with torch.no_grad():
        print("embedding", tuple(m(torch.randn(2, 4 * 16000) * 0.01).shape))
