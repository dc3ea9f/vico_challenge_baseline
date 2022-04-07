from bisect import bisect_right
from math import ceil
import numpy as np
import os
import os.path as osp
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.distributed as dist
from copy import deepcopy
from box import Box
import torch.nn.utils.rnn as rnn_utils


FEATURE_DIR = 'features'
METADATA_DIR = 'metadata'
AUDIO_FEATS_DIR = 'audio_feats'
VIDEO_FEATS_DIR = 'video_feats'
class VicoDataset(data.Dataset):
    def __init__(self, root, task, dynam_3dmm_transform=None, audio_transform=None):
        self.root = root
        assert task in ['listener', 'speaker']
        self.task = task
        self.video_root = osp.join(self.root, FEATURE_DIR, VIDEO_FEATS_DIR)
        self.audio_root = osp.join(self.root, FEATURE_DIR, AUDIO_FEATS_DIR)

        anno_file = osp.join(self.root, METADATA_DIR, 'data.csv')
        self.anno_df = pd.read_csv(anno_file)
        self.anno_df = self.anno_df.reset_index()

        raw_data = {}
        for row_idx, row in self.anno_df.iterrows():
            audio = np.load(f'{self.audio_root}/{row.uuid}.npy')
            speaker = torch.load(f'{self.video_root}/{row.uuid}.speaker.bin')
            listener = torch.load(f'{self.video_root}/{row.uuid}.listener.bin')
            raw_data[row_idx] = {
                'audio': audio,
                'speaker': speaker,
                'listener': listener,
            }
        self.data = raw_data

        self.dynam_3dmm_transform = dynam_3dmm_transform
        self.audio_transform = audio_transform

    def __getitem__(self, idx):
        row = self.anno_df.iloc[idx]

        audio = torch.from_numpy(self.data[idx]['audio'])
        speaker_video, listener_video = self.data[idx]['speaker'], self.data[idx]['listener']

        listener_3dmm_fixed, listener_3dmm_dynam = self.load_3dmm(listener_video)
        speaker_3dmm_fixed,  speaker_3dmm_dynam  = self.load_3dmm(speaker_video)

        if self.task == 'listener':
            driven_3dmm_dynam = speaker_3dmm_dynam
            init_3dmm_dynam, target_3dmm_dynam = torch.split(listener_3dmm_dynam, [1, listener_3dmm_dynam.size(0) - 1])
        else:
            driven_3dmm_dynam = listener_3dmm_dynam
            init_3dmm_dynam, target_3dmm_dynam = torch.split(speaker_3dmm_dynam, [1, speaker_3dmm_dynam.size(0) - 1])
        
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)

        if self.dynam_3dmm_transform is not None:
            driven_3dmm_dynam = self.dynam_3dmm_transform(driven_3dmm_dynam)
            init_3dmm_dynam = self.dynam_3dmm_transform(init_3dmm_dynam)
            target_3dmm_dynam = self.dynam_3dmm_transform(target_3dmm_dynam)

        driven_signal = driven_3dmm_dynam
        init_signal = init_3dmm_dynam
        target_signal = target_3dmm_dynam

        return audio, driven_signal, init_signal, target_signal, row

    def load_3dmm(self, data):
        id_gamma_tex    = torch.from_numpy(data['id.gamma.tex']).float()
        angle_exp_trans = torch.from_numpy(data['angle.exp.trans']).float()
        return id_gamma_tex, angle_exp_trans

    def __len__(self):
        return len(self.anno_df)

def get_dataset(config, task):
    mfcc_mean_std_info = torch.load(osp.join(config.root, FEATURE_DIR, 'audio_feats_mean_std.bin'))
    mfcc_mean, mfcc_std = mfcc_mean_std_info['mean'], mfcc_mean_std_info['std']
    audio_transform = transforms.Lambda(lambda e: (e - mfcc_mean) / mfcc_std)

    dynam_mean_std_info = torch.load(osp.join(config.root, FEATURE_DIR, 'video_feats_mean_std.bin'))['angle.exp.trans']
    dynam_mean, dynam_std = dynam_mean_std_info['mean'], dynam_mean_std_info['std']
    dynam_3dmm_transform = transforms.Lambda(lambda e: (e - dynam_mean) / dynam_std)

    dataset = VicoDataset(
        config['root'],
        task,
        dynam_3dmm_transform=dynam_3dmm_transform,
        audio_transform=audio_transform,
    )
    return dataset

def collate_fn(batch):
    audio = [e[0] for e in batch]
    driven = [e[1] for e in batch]
    init = [e[2] for e in batch]
    target = [e[3] for e in batch]
    rows = [e[4] for e in batch]
    lengths = torch.from_numpy(np.array([e.size(0) for e in driven]))

    audio = rnn_utils.pad_sequence(audio, batch_first=True)
    driven = rnn_utils.pad_sequence(driven, batch_first=True)
    target = rnn_utils.pad_sequence(target, batch_first=True)
    init = torch.vstack(init)
    return audio, driven, init, target, lengths, rows

def get_data_loader(config, task):
    dataset = get_dataset(config, task)
    loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=config['batch_size'],
        drop_last=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    return loader
