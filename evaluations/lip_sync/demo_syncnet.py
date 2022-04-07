#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess
from glob import glob
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

from SyncNetInstance import *

# ==================== LOAD PARAMS ====================


parser = argparse.ArgumentParser(description = "SyncNet");

parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--tmp_dir', type=str, default="./tmp", help='');
parser.add_argument('--reference', type=str, default="demo", help='');
parser.add_argument('--audio_dir', type=str, default='../../../data/listening_head/audios/', help='')
parser.add_argument('--pd_video_folder', type=str, required=True);
parser.add_argument('--anno_file', type=str, required=True)

opt = parser.parse_args();


# ==================== RUN EVALUATION ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);
print("Model %s loaded."%opt.initial_model);

if os.path.exists(os.path.join(opt.tmp_dir,opt.reference)):
    rmtree(os.path.join(opt.tmp_dir,opt.reference))
os.makedirs(os.path.join(opt.tmp_dir,opt.reference))

df = pd.read_csv(opt.anno_file)
offset_and_confs = []
dists = []
for row_idx, row in tqdm(df.iterrows(), total=len(df)):
    if row_idx > 10:
        continue
    pd_video_fn = f'{opt.pd_video_folder}/{row.uuid}.speaker.mp4'
    assert os.path.exists(pd_video_fn), f"'{pd_video_fn}' is not exist"
    offset, conf, dist = s.evaluate(opt, videofile=pd_video_fn)
    offset_and_confs.append([offset, conf])
    dists.append(dist)
print(np.array(offset_and_confs))
