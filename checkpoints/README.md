# Checkpoints

Layout, as written by `run_training.py` (`<objective>`: `protofix`, `angleproto`, `normface`, `aamsoftmax`,
`amsoftmax`, `ge2e`, `supcon`):

```
checkpoints/
  ecapa_tdnn/<objective>_s<seed>/model/model0000000NN.model
  fast_resnet34/<objective>_s<seed>/model/model0000000NN.model
  redimnet_b2/<objective>_s<seed>/model/model0000000NN.model
  wavlm/phase1/<objective>_s<seed>/model/model0000000NN.model
  wavlm/phase2/<objective>_s<seed>/model/model0000000NN.model
  ecapa_bottleneck/<objective>_d<n_out>_s<seed>/model/model0000000NN.model
```
