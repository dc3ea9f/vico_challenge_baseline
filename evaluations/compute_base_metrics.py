import pandas as pd
from glob import glob
import argparse
import cv2
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
from piq import psnr, ssim


def read_mp4(input_fn, to_rgb=False, to_gray=False, to_nchw=False):
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
        frames.append(frame)
    cap.release()
    frames = np.stack(frames)
    if to_nchw:
        frames = np.transpose(frames, (0, 3, 1, 2))
    return frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_video_folder', type=str, required=True)
    parser.add_argument('--pd_video_folder', type=str, required=True)
    parser.add_argument('--task', type=str, required=True, choices=['speaker', 'listener'])
    parser.add_argument('--anno_file', type=str, required=True)
    args = parser.parse_args()

    psnr_values = []
    ssim_values = []
    df = pd.read_csv(args.anno_file)
    for row_idx, row in tqdm(df.iterrows(), total=len(df)):
        gt_video_fn = f'{args.gt_video_folder}/{row.uuid}.{args.task}.mp4'
        pd_video_fn = f'{args.pd_video_folder}/{row.uuid}.{args.task}.mp4'

        assert osp.exists(gt_video_fn), f"'{gt_video_fn}' is not exist"
        assert osp.exists(pd_video_fn), f"'{pd_video_fn}' is not exist"

        gt_frames = read_mp4(gt_video_fn, True, False, True)
        pd_frames = read_mp4(pd_video_fn, True, False, True)

        gt_frames = torch.from_numpy(gt_frames).float() / 255.
        pd_frames = torch.from_numpy(pd_frames).float() / 255.

        psnr_value = psnr(pd_frames, gt_frames, reduction='none')
        psnr_values.extend([e.item() for e in psnr_value])

        ssim_value = ssim(pd_frames, gt_frames, data_range=1., reduction='none')
        ssim_values.extend([e.item() for e in ssim_value])
    print('psnr:', np.array(psnr_values).mean())
    print('ssim:', np.array(ssim_values).mean())
