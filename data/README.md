# Data

The corpora are obtained from their owners. Default layout (each location can be set with the
`DATA_ROOT`, `VOX1_WAV`, `VOX2_WAV`, `FEAT_SHARDS`, `VOXSIM_DIR` environment variables):

```
data/
  lists/veri_test2.txt           VoxCeleb1-O trial list
  lists/vox2_dev_train.txt.gz    VoxCeleb2-dev training list
  vox1_wav/                      VoxCeleb1 wav, id1xxxx/<video>/<utt>.wav
  feat_shards_80/                VoxCeleb2-dev log-mel, 80 bins
  feat_shards_72/                VoxCeleb2-dev log-mel, 72 bins
  vox2_wav/                      VoxCeleb2-dev 16 kHz wav
  voxsim/                        VoxSim ratings
```

## VoxCeleb2-dev

```
gunzip -k data/lists/vox2_dev_train.txt.gz
sed 's/\.npy$/.wav/' data/lists/vox2_dev_train.txt > data/lists/vox2_dev_train_wav.txt

python data/prep_vox2.py unzip   --aac_dir <dir with vox2_dev_aac_part*> --zip /scratch/vox2_dev_aac.zip --extract /scratch/vox2
python data/prep_vox2.py meldual --aac /scratch/vox2/dev/aac --out80 data/feat_shards_80 --out72 data/feat_shards_72 --workers 32
python data/prep_vox2.py wav     --aac /scratch/vox2/dev/aac --out data/vox2_wav --workers 32
```

## VoxSim

```
git clone https://github.com/kaistmm/voxsim_trainer
python data/build_voxsim_csv.py --voxsim_trainer voxsim_trainer --out data/voxsim
```

## Licenses

The two lists in `lists/` are VoxCeleb metadata (https://www.robots.ox.ac.uk/~vgg/data/voxceleb/, Visual
Geometry Group, University of Oxford, CC BY-SA 4.0): `veri_test2.txt` is the official file, and
`vox2_dev_train.txt` is the official `train_list.txt` with the `.wav` suffix replaced by `.npy` and the lines
sorted. VoxSim is from Ahn et al. (Interspeech 2024).
