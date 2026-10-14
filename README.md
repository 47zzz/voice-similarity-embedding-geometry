# Rethinking Automated Voice Similarity by Shifting from EER to Embedding Geometry

Code, configurations and seeds for

> Szu-Chi Chen, Jia-Kai Dong, Yi-Cheng Lin, Sung-Feng Huang, Hung-yi Lee.
> *Rethinking Automated Voice Similarity by Shifting from EER to Embedding Geometry.* Submitted to ICASSP 2027.

| path | content |
|---|---|
| `train/` | training code: four backbones (`models/`) and seven objectives (`loss/`) |
| `configs/` | the five model conditions, the objectives, the seeds, and the embedding-dimension sweep |
| `run_training.py` | launches the runs of one condition |
| `data/` | data preparation and lists |
| `eval/` | embedding extraction and the measures: EER, ρ_align, d_eff |
| `checkpoints/` | checkpoint layout |

## Setup

```
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -r requirements-public.txt      # public checkpoints only
```

Data: see `data/README.md`.

## Training

```
python run_training.py --config ecapa_tdnn
python run_training.py --config fast_resnet34
python run_training.py --config redimnet_b2
python run_training.py --config wavlm --phase phase1
python run_training.py --config wavlm --phase phase2
python run_training.py --config ecapa_bottleneck --nout 32      # 512, 32, 10, 3
```

Each command trains every objective with seeds 10, 11, 12; `--losses` and `--seeds` select a subset.
Runs are written to `checkpoints/`.

## Checkpoints

The final-epoch checkpoint of seed 10 for every configuration:
https://huggingface.co/47z/voice-similarity-embedding-geometry

## Evaluation

Evaluation needs the VoxCeleb1 audio and the VoxSim ratings (`data/README.md`).

```
# 1. download a checkpoint (without --include: all checkpoints)
hf download 47z/voice-similarity-embedding-geometry --include "ecapa_tdnn/angleproto_s10/*" --local-dir checkpoints

# 2. inference: embeddings of the VoxSim and VoxCeleb1-O clips
python eval/extract_embeddings.py --config ecapa_tdnn \
    --ckpt checkpoints/ecapa_tdnn/angleproto_s10/model/*.model \
    --out embeddings/ecapa_tdnn/angleproto_s10

# 3. EER, rho_align, d_eff
python eval/score.py --emb_dir embeddings/ecapa_tdnn/angleproto_s10 --name ecapa_tdnn/angleproto/s10 --append scores.csv
```

`--config` is the top-level folder of the checkpoint (`wavlm` for both WavLM phases); for `ecapa_bottleneck`
also pass the width of the run, e.g. `--nout 32` for `amsoftmax_d32_s10`.

Public checkpoints: `eval/public/extract_public.py`, then the same `eval/score.py`.

## License

MIT (`LICENSE`). Third-party components keep their own licenses: `train/NOTICE.md`, `data/README.md`.

## Citation

```
@inproceedings{chen2027geometry,
  title     = {Rethinking Automated Voice Similarity by Shifting from {EER} to Embedding Geometry},
  author    = {Chen, Szu-Chi and Dong, Jia-Kai and Lin, Yi-Cheng and Huang, Sung-Feng and Lee, Hung-yi},
  booktitle = {submitted to ICASSP 2027},
  year      = {2027}
}
```
