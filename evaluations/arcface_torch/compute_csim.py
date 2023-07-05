import argparse
import os.path as osp
import pandas as pd
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from backbones import get_model


def read_mp4(input_fn, target_size=None, to_rgb=False, to_gray=False, to_nchw=False):
    frames = []
    cap = cv2.VideoCapture(input_fn)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if target_size is not None:
            frame = cv2.resize(frame, target_size)
        frames.append(frame)
    cap.release()
    frames = np.stack(frames)
    if to_nchw:
        frames = np.transpose(frames, (0, 3, 1, 2))
    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_video_folder', type=str, required=True)
    parser.add_argument('--pd_video_folder', type=str, required=True)
    parser.add_argument('--task', type=str, required=True, choices=['speaker', 'listener'])
    parser.add_argument('--anno_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--weight', type=str, required=True)
    args = parser.parse_args()

    net = get_model('r100', fp16=False)
    net.load_state_dict(torch.load(args.weight))
    net = net.cuda()
    net.eval()

    gt_feats = []
    pd_feats = []
    df = pd.read_csv(args.anno_file)
    for row_idx, row in tqdm(df.iterrows(), total=len(df)):
        gt_video_fn = f'{args.gt_video_folder}/{row.uuid}.{args.task}.mp4'
        pd_video_fn = f'{args.pd_video_folder}/{row.uuid}.{args.task}.mp4'

        assert osp.exists(gt_video_fn), f"'{gt_video_fn}' is not exist"
        assert osp.exists(pd_video_fn), f"'{pd_video_fn}' is not exist"

        gt_frames = read_mp4(gt_video_fn, (112, 112), True, False, True)
        pd_frames = read_mp4(pd_video_fn, (112, 112), True, False, True)

        gt_frames = torch.from_numpy(gt_frames).float()
        pd_frames = torch.from_numpy(pd_frames).float()

        gt_frames.div_(255).sub_(0.5).div_(0.5)
        pd_frames.div_(255).sub_(0.5).div_(0.5)

        T = gt_frames.size(0)
        total_images = torch.cat((gt_frames, pd_frames), 0).cuda()
        if len(total_images) > args.batch_size:
            total_images = torch.split(total_images, args.batch_size, 0)
        else:
            total_images = [total_images]

        total_feats = []
        for sub_images in total_images:
            with torch.no_grad():
                feats = net(sub_images)
            feats = feats.detach().cpu()
            total_feats.append(feats)
        total_feats = torch.cat(total_feats, 0)
        gt_feat, pd_feat = torch.split(total_feats, (T, T), 0)

        gt_feats.append(gt_feat.numpy())
        pd_feats.append(pd_feat.numpy())
    gt_feats = torch.from_numpy(np.concatenate(gt_feats, 0))
    pd_feats = torch.from_numpy(np.concatenate(pd_feats, 0))
    print('cosine similarity:', F.cosine_similarity(gt_feats, pd_feats).mean().item())
