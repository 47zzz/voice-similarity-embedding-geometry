import torch
import numpy
import random
import json
import soundfile
from pathlib import Path
from torch.utils.data import Dataset


def round_down(num, divisor):
    return num - (num % divisor)


def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)


def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

    max_audio = max_frames * 160 + 240

    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1
        audio       = numpy.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random() * (audiosize - max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = numpy.stack(feats, axis=0).astype(numpy.float64)

    return feat


class train_dataset_loader(Dataset):

    def __init__(self, train_list, max_frames, train_path, **kwargs):

        self.train_list = train_list
        self.max_frames = max_frames

        with open(train_list) as dataset_file:
            lines = dataset_file.readlines()

        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        self.data_list  = []
        self.data_label = []

        for lidx, line in enumerate(lines):
            data = line.strip().split()

            speaker_label = dictkeys[data[0]]
            filename = str(Path(train_path) / data[1])

            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, indices):

        feat = []

        for index in indices:
            audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False)
            feat.append(audio)

        feat = numpy.concatenate(feat, axis=0)

        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class test_dataset_loader(Dataset):
    def __init__(self, test_list, test_path, eval_frames, num_eval, **kwargs):
        self.max_frames = eval_frames
        self.num_eval   = num_eval
        self.test_path  = test_path
        self.test_list  = test_list

    def __getitem__(self, index):
        audio = loadWAV(str(Path(self.test_path) / self.test_list[index]), self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)


class train_dataset_sampler(torch.utils.data.Sampler):

    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, seed, **kwargs):

        self.data_label         = data_source.data_label
        self.nPerSpeaker        = nPerSpeaker
        self.max_seg_per_spk    = max_seg_per_spk
        self.batch_size         = batch_size
        self.epoch              = 0
        self.seed               = seed
        self.num_samples        = 0

    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()

        data_dict = {}

        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = []
            data_dict[speaker_label].append(index)

        dictkeys = list(data_dict.keys())
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        for findex, key in enumerate(dictkeys):
            data    = data_dict[key]
            numSeg  = round_down(min(len(data), self.max_seg_per_spk), self.nPerSpeaker)

            rp      = lol(numpy.arange(numSeg), self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        mixid           = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel        = []
        mixmap          = []

        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        mixed_list = [flattened_list[i] for i in mixmap]

        total_size = round_down(len(mixed_list), self.batch_size)
        self.num_samples = total_size
        return iter(mixed_list[:total_size])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class feat_shard_dataset_loader(Dataset):

    def __init__(self, train_list, feat_shard_path, max_frames, train_path=None, **kwargs):
        self.max_frames = max_frames
        self.root = Path(feat_shard_path)
        with open(train_list) as f:
            lines = f.readlines()
        dictkeys = {k: i for i, k in enumerate(sorted(set(x.split()[0] for x in lines)))}
        idx_cache = {}
        self.data_list, self.data_label = [], []
        for line in lines:
            spk, rel = line.strip().split()[:2]
            rel = rel.rsplit('.', 1)[0]
            if spk not in idx_cache:
                idx_cache[spk] = json.loads((self.root / f"{spk}.idx.json").read_text(encoding="utf-8"))
            s, e = idx_cache[spk][rel]
            self.data_label.append(dictkeys[spk])
            self.data_list.append((str(self.root / f"{spk}.npy"), s, e))

    def __getitem__(self, indices):
        feat = []
        for index in indices:
            shard, s, e = self.data_list[index]
            mm = numpy.load(shard, mmap_mode='r')
            mel = numpy.asarray(mm[:, s:e])
            T = mel.shape[1]
            if T <= self.max_frames:
                mel = numpy.pad(mel, ((0, 0), (0, self.max_frames - T + 1)), 'wrap')
                T = mel.shape[1]
            st = numpy.int64(random.random() * (T - self.max_frames))
            feat.append(mel[numpy.newaxis, :, st:st + self.max_frames])
        feat = numpy.concatenate(feat, axis=0).astype(numpy.float32)
        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)
